#!/usr/bin/env python3
"""
Test script for RDF generation with specific problematic cases.

This allows iterating on prompt changes without running the full pipeline.
"""

import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from schema_matcher import SchemaMatcher
from src.rdf_tools import create_rdf_tools, triples_to_turtle
from src.prompts import RDF_STATEMENT_SYSTEM_PROMPT, RDF_STATEMENT_HUMAN_PROMPT

# Configuration - match the notebook
LLM_CONFIG = {
    "model": "qwen/qwen3-coder-30b",
    "temperature": 1,
    "base_url": os.environ.get("LM_STUDIO_BASE_URL", "http://host.docker.internal:1234/v1"),
}
VOCAB_CACHE_DIR = "data/vocab_cache"

# Test cases - problematic statements that should use Role pattern
TEST_CASES = [
    {
        "id": "3.1",
        "statement": "[Albert Einstein](/wiki/Albert_Einstein) worked at the [University of Zurich](/wiki/University_of_Zurich) from 1909 to 1911.",
        "expected_pattern": "OrganizationRole with startDate/endDate, blank node _:s3_1_..."
    },
    {
        "id": "3.2", 
        "statement": "[Albert Einstein](/wiki/Albert_Einstein) received the [Gold Medal of RAS](/wiki/Gold_Medal_of_the_Royal_Astronomical_Society) in 1926.",
        "expected_pattern": "Role with date qualifier for award"
    },
    {
        "id": "3.3",
        "statement": "[Albert Einstein](/wiki/Albert_Einstein) attended St. Peter's [Catholic elementary school](/wiki/Catholic_school) in Munich.",
        "expected_pattern": "schema:alumniOf with OrganizationRole (education)"
    },
    {
        "id": "3.4",
        "statement": "[Albert Einstein](/wiki/Albert_Einstein) objected that \"God does not play dice\".",
        "expected_pattern": "schema:Quotation with text and creator - NOT rdfs:comment or schema:description"
    },
    {
        "id": "3.5",
        "statement": "[Albert Einstein](/wiki/Albert_Einstein) was five years old when he was sick in bed.",
        "expected_pattern": "Skip or rdfs:comment - no good schema match, NOT birthDate"
    },
    {
        "id": "6.1",
        "statement": "In January 1896, [Albert Einstein](/wiki/Albert_Einstein) renounced his [citizenship of the German Kingdom of Württemberg](/wiki/German_citizenship) in order to avoid [conscription into military service](/wiki/Conscription_in_Germany).",
        "expected_pattern": "Role with endDate for renounced citizenship, NOT simple nationality"
    },
]

def setup_matcher():
    """Initialize the schema matcher from cache (same as notebook)."""
    matcher = SchemaMatcher.load(VOCAB_CACHE_DIR, embed_base_url=LLM_CONFIG["base_url"])
    print(f"Loaded schema matcher with {len(matcher.vocabularies)} vocabularies")
    return matcher

def run_rdf_generation(matcher, statements, verbose=True):
    """Run RDF generation on a set of statements."""
    
    # Create tools
    rdf_tools, get_triples, reset_triples = create_rdf_tools(matcher)
    reset_triples()
    
    # Setup LLM - same config as notebook
    rdf_llm = ChatOpenAI(
        model=LLM_CONFIG["model"],
        temperature=LLM_CONFIG["temperature"],
        base_url=LLM_CONFIG["base_url"],
        api_key="lm-studio",
        timeout=60,
        max_retries=0,
    )
    rdf_llm_with_tools = rdf_llm.bind_tools(rdf_tools)
    
    # Build tools dict for execution
    tools_by_name = {tool.name: tool for tool in rdf_tools}
    
    # Format statements
    statements_text = "\n".join([
        f"[{s['id']}] {s['statement']}" for s in statements
    ])
    
    # Entity registry (minimal for testing)
    entity_registry = """- Albert Einstein: <https://en.wikipedia.org/wiki/Albert_Einstein>"""
    
    # Create messages
    messages = [
        SystemMessage(content=RDF_STATEMENT_SYSTEM_PROMPT),
        HumanMessage(content=RDF_STATEMENT_HUMAN_PROMPT.format(
            source_url="https://en.wikipedia.org/wiki/Albert_Einstein",
            breadcrumb="Test",
            entity_registry=entity_registry,
            statements=statements_text
        ))
    ]
    
    if verbose:
        print("\n" + "="*60)
        print("STATEMENTS:")
        print(statements_text)
        print("="*60)
    
    # Run with tool calling loop
    max_iterations = 10
    for iteration in range(max_iterations):
        response = rdf_llm_with_tools.invoke(messages)
        
        if verbose:
            print(f"\n--- Iteration {iteration + 1} ---")
        
        if hasattr(response, 'tool_calls') and response.tool_calls:
            if verbose:
                print(f"Tool calls: {len(response.tool_calls)}")
            
            messages.append(response)
            
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                
                if verbose and tool_name in ("emit_triple", "emit_triples"):
                    print(f"  {tool_name}: {tool_args}")
                
                tool = tools_by_name.get(tool_name)
                if tool:
                    result = tool.invoke(tool_args)
                    messages.append(ToolMessage(
                        content=str(result),
                        tool_call_id=tool_call["id"]
                    ))
        else:
            # No more tool calls - done
            if verbose:
                print(f"\nFinal response: {response.content[:200]}...")
            break
    
    # Get results
    triples = get_triples()
    turtle = triples_to_turtle(triples)
    
    return triples, turtle


def analyze_results(statements, triples):
    """Analyze whether the results match expected patterns."""
    print("\n" + "="*60)
    print("ANALYSIS:")
    print("="*60)
    
    for stmt in statements:
        stmt_id = stmt["id"]
        # Handle dotted IDs like "3.1"
        stmt_triples = [t for t in triples if t["statement_id"] == stmt_id or t["statement_id"] == stmt_id.replace(".", "_")]
        
        print(f"\n[{stmt_id}] {stmt['statement'][:60]}...")
        print(f"    Expected: {stmt['expected_pattern']}")
        print(f"    Got {len(stmt_triples)} triples:")
        
        # Check for Role pattern
        has_role = any("Role" in t["object"] for t in stmt_triples)
        has_blank_node = any(t["subject"].startswith("_:") or t["object"].startswith("_:") for t in stmt_triples)
        has_dates = any("Date" in t["predicate"] or "date" in t["predicate"].lower() for t in stmt_triples)
        has_description_misuse = any(t["predicate"] == "schema:description" for t in stmt_triples)
        has_comment = any("comment" in t["predicate"].lower() for t in stmt_triples)
        has_quotation = any("Quotation" in t["object"] for t in stmt_triples)
        has_text = any("schema:text" in t["predicate"] for t in stmt_triples)
        
        for t in stmt_triples:
            print(f"      {t['subject']} {t['predicate']} {t['object']}")
        
        if len(stmt_triples) == 0:
            print(f"    ⚠ Skipped (no triples emitted)")
        elif has_role or has_blank_node:
            print(f"    ✓ Uses Role/blank node pattern")
        
        if has_dates:
            print(f"    ✓ Has date predicates")
        
        if has_description_misuse:
            print(f"    ✗ PROBLEM: Misused schema:description!")
        
        if has_comment:
            print(f"    ✓ Used rdfs:comment for unstructured info")
        
        if has_quotation:
            print(f"    ✓ Used schema:Quotation for quote!")
        
        if has_text:
            print(f"    ✓ Used schema:text for quote content!")


def main():
    print("Setting up schema matcher...")
    matcher = setup_matcher()
    
    print("\nRunning RDF generation on test cases...")
    triples, turtle = run_rdf_generation(matcher, TEST_CASES, verbose=True)
    
    print("\n" + "="*60)
    print("GENERATED TURTLE:")
    print("="*60)
    print(turtle)
    
    analyze_results(TEST_CASES, triples)


if __name__ == "__main__":
    main()
