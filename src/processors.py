"""Pipeline processing functions for facts extraction and RDF generation."""

import json
import re
import time
from typing import List

import nbformat
from nbformat.v4 import new_markdown_cell, new_raw_cell
from langchain_core.messages import ToolMessage
from langchain_core.prompts import ChatPromptTemplate

from .cid import compute_cid, make_signature, parse_signature, extract_signatures, extract_statement_signatures
from .prompts import RDF_STATEMENT_SYSTEM_PROMPT, RDF_STATEMENT_HUMAN_PROMPT
from .rdf_tools import parse_statements
from .utils import log_progress


def process_facts_extraction(
    chunk_data: list[dict],
    facts_nb: nbformat.NotebookNode,
    facts_signatures: dict,
    facts_chain,
    provenance: dict,
    registry,
    facts_path: str,
    timeout_seconds: int
) -> tuple[int, int, int]:
    """Process all chunks to extract facts.
    
    Returns:
        tuple of (processed_count, skipped_count, error_count)
    """
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    for chunk in chunk_data:
        cell_num = chunk["cell_num"]
        source_cid = chunk["cid"]
        
        # Check if we already have up-to-date facts for this chunk
        existing_sig = facts_signatures.get(cell_num)
        if existing_sig and existing_sig["from_cid"] == source_cid:
            log_progress(f"  Chunk {cell_num}: ⊘ Up-to-date (CID match), skipping")
            skipped_count += 1
            continue
        
        # Need to generate (or regenerate) this cell
        status = "↻ Regenerating" if existing_sig else "+ Generating"
        chunk_preview = chunk["chunk_text"][:80].replace('\n', ' ')
        log_progress(f"  Chunk {cell_num}: {status}")
        log_progress(f"    Context: {chunk['breadcrumb']}")
        log_progress(f"    Input: {chunk_preview}...")
        log_progress(f"    [Calling LLM...]", end=" ")
        
        start_time = time.time()
        
        # Call LLM to extract facts
        try:
            result = facts_chain.invoke({
                "source_url": provenance["source_url"],
                "breadcrumb": chunk["breadcrumb"],
                "known_entities": registry.get_known_entities_text(),
                "chunk_text": chunk["chunk_text"],
            })
            facts_content = result.content
            elapsed = time.time() - start_time
            log_progress(f"✓ ({len(facts_content)} chars, {elapsed:.1f}s)")
            processed_count += 1
        except Exception as e:
            elapsed = time.time() - start_time
            error_type = type(e).__name__
            error_msg = str(e)[:100]
            facts_content = f"# Error: {error_type}: {e}"
            log_progress(f"✗ {error_type} after {elapsed:.1f}s")
            log_progress(f"    Error: {error_msg}")
            error_count += 1
        
        # Build the facts cell content
        facts_cell_content = f"""**Context:** {chunk['breadcrumb']}
**Chunk:** {cell_num} of {len(chunk_data)}

---

{facts_content}
"""
        facts_cid = compute_cid(facts_cell_content)
        signature = make_signature(cell_num, "facts", facts_cid, source_cid)
        
        # Remove old cells if regenerating
        if existing_sig:
            new_cells = [facts_nb.cells[0], facts_nb.cells[1]]  # Keep header
            i = 2
            while i < len(facts_nb.cells):
                cell = facts_nb.cells[i]
                if cell.cell_type == 'raw':
                    sig = parse_signature(cell.source)
                    if sig and sig["cell"] == cell_num:
                        i += 1
                        continue
                if i > 0 and i + 1 < len(facts_nb.cells):
                    next_sig = parse_signature(facts_nb.cells[i + 1].source) if facts_nb.cells[i + 1].cell_type == 'raw' else None
                    if next_sig and next_sig["cell"] == cell_num:
                        i += 2
                        continue
                new_cells.append(cell)
                i += 1
            facts_nb.cells = new_cells
        
        # Append new content and signature
        facts_nb.cells.append(new_markdown_cell(facts_cell_content))
        facts_nb.cells.append(new_raw_cell(json.dumps(signature)))
        
        # Update signatures dict
        facts_signatures[cell_num] = signature
        
        # Save notebook after each cell
        log_progress(f"    [Saving notebook...]", end=" ")
        with open(facts_path, 'w', encoding='utf-8') as f:
            nbformat.write(facts_nb, f)
        log_progress("saved")
    
    return processed_count, skipped_count, error_count


def call_llm_with_tools(
    prompt_vars: dict,
    rdf_prompt: ChatPromptTemplate,
    rdf_llm_with_tools,
    rdf_tools: list,
    get_triples_fn,
    reset_triples_fn,
    max_iterations: int
) -> tuple[str, List[dict], int, List[dict]]:
    """Call LLM with tool support. Returns (summary_response, collected_triples, iterations_used, tool_calls_log)."""
    reset_triples_fn()  # Reset collector
    
    messages = rdf_prompt.format_messages(**prompt_vars)
    
    # Build tool lookup
    tool_map = {t.name: t for t in rdf_tools}
    
    # Track all tool calls for debugging
    tool_calls_log = []
    
    response = None
    for iteration in range(max_iterations):
        response = rdf_llm_with_tools.invoke(messages)
        
        # Check if there are tool calls
        if hasattr(response, 'tool_calls') and response.tool_calls:
            # Add assistant message with tool calls
            messages.append(response)
            
            # Execute each tool call
            for tool_call in response.tool_calls:
                tool_name = tool_call['name']
                tool_args = tool_call['args']
                
                # Log the tool call
                tool_calls_log.append({
                    "iteration": iteration + 1,
                    "name": tool_name,
                    "args": tool_args,
                })
                
                # Find and execute the tool
                if tool_name in tool_map:
                    try:
                        tool_result = tool_map[tool_name].invoke(tool_args)
                    except TypeError as e:
                        # Missing/wrong arguments - give specific guidance
                        tool_result = f"INVALID CALL to {tool_name}: {e}. Check the required arguments and try again."
                        tool_calls_log[-1]["error"] = str(e)
                    except Exception as e:
                        tool_result = f"ERROR in {tool_name}: {type(e).__name__}: {e}. You may retry with corrected input."
                        tool_calls_log[-1]["error"] = str(e)
                else:
                    tool_result = f"Unknown tool: {tool_name}. Available: emit_triple, emit_triples, find_rdf_class, find_rdf_property"
                
                tool_calls_log[-1]["result"] = str(tool_result)[:200]
                
                # Add tool result message
                messages.append(ToolMessage(content=str(tool_result), tool_call_id=tool_call['id']))
        else:
            # No tool calls, return the summary and collected triples
            summary = response.content if hasattr(response, 'content') else str(response)
            return summary, get_triples_fn(), iteration + 1, tool_calls_log
    
    # Max iterations reached
    summary = response.content if hasattr(response, 'content') else "Max iterations reached"
    return summary, get_triples_fn(), max_iterations, tool_calls_log


def process_rdf_generation(
    facts_data: list[dict],
    rdf_nb: nbformat.NotebookNode,
    rdf_signatures: dict,
    provenance: dict,
    registry,
    rdf_path: str,
    rdf_prompt: ChatPromptTemplate,
    rdf_llm_with_tools,
    rdf_tools: list,
    get_triples_fn,
    reset_triples_fn,
    max_iterations: int,
    timeout_seconds: int
) -> tuple[int, int, int, int, list[int]]:
    """Process facts by chunk to generate RDF triples.
    
    Returns:
        tuple of (processed_count, skipped_count, error_count, total_triples, iteration_counts)
    """
    processed_count = 0
    skipped_count = 0
    error_count = 0
    total_triples = 0
    iteration_counts = []
    max_iterations_hit = 0
    total_statements = 0
    
    for facts_item in facts_data:
        chunk_num = facts_item["cell_num"]
        
        if facts_item["facts_text"].startswith("# Error:"):
            log_progress(f"  Chunk {chunk_num}: ⊘ Skipping (error in facts)")
            continue
        
        # Parse statements for this chunk
        statements = parse_statements(facts_item["facts_text"])
        if not statements:
            log_progress(f"  Chunk {chunk_num}: ⊘ No statements found")
            continue
        
        total_statements += len(statements)
        
        # Build statement list with IDs and compute combined CID
        stmt_items = []
        stmt_cids = []
        needs_processing = False
        
        for idx, stmt in enumerate(statements):
            stmt_id = str(idx + 1)
            stmt_key = f"{chunk_num}_{idx + 1}"
            stmt_cid = compute_cid(stmt)
            stmt_cids.append(stmt_cid)
            
            # Check if we already have up-to-date RDF for this statement
            existing_sig = rdf_signatures.get(stmt_key)
            if not existing_sig or existing_sig.get("from_cid") != stmt_cid:
                needs_processing = True
            
            stmt_items.append({
                "id": stmt_id,
                "stmt_key": stmt_key,
                "statement": stmt,
                "cid": stmt_cid,
            })
        
        # Compute combined CID for the chunk's statements
        combined_cid = compute_cid("|".join(stmt_cids))
        
        # Check if entire chunk is up-to-date
        chunk_key = f"chunk_{chunk_num}"
        existing_chunk_sig = rdf_signatures.get(chunk_key)
        if existing_chunk_sig and existing_chunk_sig.get("from_cid") == combined_cid:
            log_progress(f"  Chunk {chunk_num}: ⊘ Up-to-date ({len(statements)} stmts)")
            skipped_count += len(statements)
            continue
        
        # Format statements for prompt
        statements_text = "\n".join([f"[{s['id']}] {s['statement']}" for s in stmt_items])
        
        log_progress(f"  Chunk {chunk_num}: + Processing {len(statements)} statements...", end=" ")
        
        start_time = time.time()
        
        try:
            summary, triples, iterations_used, tool_calls_log = call_llm_with_tools(
                prompt_vars={
                    "source_url": provenance["source_url"],
                    "breadcrumb": facts_item["breadcrumb"],
                    "entity_registry": registry.format_for_prompt(),
                    "statements": statements_text,
                },
                rdf_prompt=rdf_prompt,
                rdf_llm_with_tools=rdf_llm_with_tools,
                rdf_tools=rdf_tools,
                get_triples_fn=get_triples_fn,
                reset_triples_fn=reset_triples_fn,
                max_iterations=max_iterations
            )
            elapsed = time.time() - start_time
            triple_count = len(triples)
            total_triples += triple_count
            iteration_counts.append(iterations_used)
            
            # Check if we hit the limit
            if iterations_used >= max_iterations:
                max_iterations_hit += 1
                log_progress(f"⚠ {triple_count}t/{iterations_used}i [MAX] {elapsed:.1f}s")
            else:
                log_progress(f"✓ {triple_count}t/{iterations_used}i {elapsed:.1f}s")
            
            # Log if no triples and we had emit calls
            emit_calls = [c for c in tool_calls_log if c["name"] in ("emit_triple", "emit_triples")]
            if triple_count == 0 and emit_calls:
                log_progress(f"    WARNING: {len(emit_calls)} emit calls but 0 triples collected!")
                log_progress(f"    Prompt statements: {statements_text[:200]}...")
                for call in emit_calls[:3]:
                    log_progress(f"    - {call['name']}: {json.dumps(call['args'])[:150]}")
            
            # Group triples by statement_id (normalize for matching)
            triples_by_stmt = {}
            for t in triples:
                stmt_id = str(t.get("statement_id", "unknown")).strip()
                if stmt_id not in triples_by_stmt:
                    triples_by_stmt[stmt_id] = []
                triples_by_stmt[stmt_id].append(t)
            
            # Remove old cells for this chunk
            new_cells = [rdf_nb.cells[0], rdf_nb.cells[1]]
            i = 2
            while i < len(rdf_nb.cells):
                cell = rdf_nb.cells[i]
                if cell.cell_type == 'raw':
                    sig = parse_signature(cell.source)
                    if sig and sig.get("chunk_num") == chunk_num:
                        i += 1
                        continue
                if i + 1 < len(rdf_nb.cells) and rdf_nb.cells[i + 1].cell_type == 'raw':
                    next_sig = parse_signature(rdf_nb.cells[i + 1].source)
                    if next_sig and next_sig.get("chunk_num") == chunk_num:
                        i += 2
                        continue
                new_cells.append(cell)
                i += 1
            rdf_nb.cells = new_cells
            
            # Add cells for each statement's triples
            for stmt_item in stmt_items:
                stmt_id = stmt_item["id"]
                stmt_key = stmt_item["stmt_key"]
                statement = stmt_item["statement"]
                stmt_cid = stmt_item["cid"]
                stmt_triples = triples_by_stmt.get(stmt_id, [])
                
                # Convert triples to Turtle with statement provenance
                rdf_lines = [f"# Statement [{chunk_num}.{stmt_id}]: {statement}"]
                for t in stmt_triples:
                    rdf_lines.append(f"{t['subject']} {t['predicate']} {t['object']} .")
                if not stmt_triples:
                    rdf_lines.append("# No triples emitted for this statement")
                    # Add tool call debug info for this statement
                    stmt_emit_calls = [c for c in tool_calls_log 
                                      if c["name"] in ("emit_triple", "emit_triples")]
                    if stmt_emit_calls:
                        rdf_lines.append(f"# Debug: {len(stmt_emit_calls)} emit tool calls made for chunk")
                        for call in stmt_emit_calls[:5]:
                            args_str = json.dumps(call["args"], default=str)[:200]
                            rdf_lines.append(f"#   {call['name']}: {args_str}")
                rdf_content = "\n".join(rdf_lines)
                
                # Build signature
                rdf_cid = compute_cid(rdf_content)
                signature = {
                    "cell": len(rdf_nb.cells),
                    "stmt_key": stmt_key,
                    "chunk_num": chunk_num,
                    "stmt_idx": int(stmt_id),
                    "type": "rdf",
                    "cid": rdf_cid,
                    "from_cid": stmt_cid,
                }
                
                # Append new content and signature
                rdf_nb.cells.append(new_raw_cell(rdf_content))
                rdf_nb.cells.append(new_raw_cell(json.dumps(signature)))
                
                # Update signatures dict
                rdf_signatures[stmt_key] = signature
            
            # Add chunk-level signature for quick skip check
            rdf_signatures[chunk_key] = {"from_cid": combined_cid}
            
            processed_count += len(statements)
        except Exception as e:
            elapsed = time.time() - start_time
            error_type = type(e).__name__
            error_msg = str(e)[:200]
            log_progress(f"✗ {error_type} {elapsed:.1f}s")
            log_progress(f"    Error: {error_msg}")
            log_progress(f"    Prompt: {statements_text[:300]}...")
            
            # Add error cell for the chunk with debug info
            error_lines = [
                f"# Chunk {chunk_num} Error: {error_type}: {error_msg}",
                f"# Statements attempted:",
            ]
            for stmt_item in stmt_items:
                error_lines.append(f"#   [{stmt_item['id']}] {stmt_item['statement'][:80]}...")
            rdf_content = "\n".join(error_lines)
            rdf_nb.cells.append(new_raw_cell(rdf_content))
            
            error_count += len(statements)
        
        # Save notebook after each chunk
        with open(rdf_path, 'w', encoding='utf-8') as f:
            nbformat.write(rdf_nb, f)
    
    log_progress(f"\nTotal statements parsed: {total_statements}")
    
    # Log iteration statistics
    if iteration_counts:
        log_progress(f"\nIteration statistics (max={max_iterations}):")
        log_progress(f"  - Min: {min(iteration_counts)}")
        log_progress(f"  - Max: {max(iteration_counts)}")
        log_progress(f"  - Mean: {sum(iteration_counts)/len(iteration_counts):.1f}")
        log_progress(f"  - Hit limit: {max_iterations_hit} chunks")
        
        # Distribution buckets
        buckets = {"1-5": 0, "6-10": 0, "11-20": 0, "21-50": 0, "51-100": 0, "100+": 0}
        for count in iteration_counts:
            if count <= 5:
                buckets["1-5"] += 1
            elif count <= 10:
                buckets["6-10"] += 1
            elif count <= 20:
                buckets["11-20"] += 1
            elif count <= 50:
                buckets["21-50"] += 1
            elif count <= 100:
                buckets["51-100"] += 1
            else:
                buckets["100+"] += 1
        
        log_progress(f"  - Distribution: {buckets}")
    
    return processed_count, skipped_count, error_count, total_triples, iteration_counts


def read_chunk_data(chunks_nb: nbformat.NotebookNode) -> list[dict]:
    """Read and parse chunk data from chunks notebook."""
    chunk_data = []
    cell_idx = 2  # Skip provenance and registry
    
    while cell_idx < len(chunks_nb.cells):
        cell = chunks_nb.cells[cell_idx]
        if cell.cell_type == 'markdown':
            content = cell.source
            # Get corresponding signature (next cell)
            sig = None
            if cell_idx + 1 < len(chunks_nb.cells):
                sig = parse_signature(chunks_nb.cells[cell_idx + 1].source)
            
            # Extract breadcrumb
            context_match = re.search(r'\*\*Context:\*\*\s*(.+)', content)
            breadcrumb = context_match.group(1) if context_match else "Unknown"
            
            # Extract chunk text (after ---)
            parts = content.split("---\n", 1)
            chunk_text = parts[1].strip() if len(parts) > 1 else content
            
            chunk_data.append({
                "cell_num": sig["cell"] if sig else len(chunk_data) + 1,
                "content": content,
                "chunk_text": chunk_text,
                "breadcrumb": breadcrumb,
                "cid": sig["cid"] if sig else compute_cid(content),
            })
            cell_idx += 2  # Skip content and signature
        else:
            cell_idx += 1
    
    return chunk_data


def read_facts_data(facts_nb: nbformat.NotebookNode) -> list[dict]:
    """Read and parse facts data from facts notebook."""
    facts_data = []
    cell_idx = 2  # Skip provenance and registry
    
    while cell_idx < len(facts_nb.cells):
        cell = facts_nb.cells[cell_idx]
        if cell.cell_type == 'markdown':
            content = cell.source
            # Get corresponding signature (next cell)
            sig = None
            if cell_idx + 1 < len(facts_nb.cells):
                sig = parse_signature(facts_nb.cells[cell_idx + 1].source)
            
            # Extract breadcrumb
            context_match = re.search(r'\*\*Context:\*\*\s*(.+)', content)
            breadcrumb = context_match.group(1) if context_match else "Unknown"
            
            # Extract facts (after ---)
            parts = content.split("---\n", 1)
            facts_text = parts[1].strip() if len(parts) > 1 else content
            
            facts_data.append({
                "cell_num": sig["cell"] if sig else len(facts_data) + 1,
                "content": content,
                "facts_text": facts_text,
                "breadcrumb": breadcrumb,
                "cid": sig["cid"] if sig else compute_cid(content),
            })
            cell_idx += 2  # Skip content and signature
        else:
            cell_idx += 1
    
    return facts_data
