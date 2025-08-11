#!/usr/bin/env python3
"""
Quality Assurance validation script for multi-dimensional vector tasks.
Manual validation of the three tasks in review status.
"""

import sys
import os
sys.path.append('/root/Archon-V2-Alpha/python/src')

def test_dimension_column_mapping():
    """Test the dimension to column name mapping utility."""
    print("=== Testing Dimension Column Mapping ===")
    
    try:
        from server.services.embeddings.embedding_service import get_dimension_column_name
        
        test_cases = [
            (768, "embedding_768"),
            (1024, "embedding_1024"), 
            (1536, "embedding_1536"),
            (3072, "embedding_3072"),
            (999, "embedding_1536")  # Unsupported should fallback
        ]
        
        all_passed = True
        for dims, expected in test_cases:
            result = get_dimension_column_name(dims)
            if result == expected:
                print(f"✅ {dims} dimensions -> {result}")
            else:
                print(f"❌ {dims} dimensions -> {result} (expected {expected})")
                all_passed = False
                
        return all_passed
        
    except Exception as e:
        print(f"❌ Error importing or testing dimension mapping: {e}")
        return False


def test_vector_search_rpc_params():
    """Test vector search RPC parameter building."""
    print("\n=== Testing Vector Search RPC Parameters ===")
    
    try:
        from server.services.search.vector_search_service import build_rpc_params
        
        test_cases = [
            ([0.1] * 768, "query_embedding_768"),
            ([0.2] * 1024, "query_embedding_1024"),
            ([0.3] * 1536, "query_embedding_1536"),
            ([0.4] * 3072, "query_embedding_3072")
        ]
        
        all_passed = True
        for embedding, expected_param in test_cases:
            params = build_rpc_params(embedding, match_count=5)
            
            if expected_param in params:
                print(f"✅ {len(embedding)} dims -> {expected_param} parameter")
                if params[expected_param] == embedding:
                    print(f"   ✅ Embedding data correctly passed")
                else:
                    print(f"   ❌ Embedding data mismatch")
                    all_passed = False
            else:
                print(f"❌ {len(embedding)} dims -> Missing {expected_param}")
                print(f"   Available params: {list(params.keys())}")
                all_passed = False
        
        # Test error handling
        try:
            params = build_rpc_params(None, match_count=5)
            if "query_embedding_1536" in params:
                print("✅ Error handling: None embedding falls back to 1536")
            else:
                print("❌ Error handling: Failed to fallback on None embedding")
                all_passed = False
        except Exception as e:
            print(f"❌ Error handling failed: {e}")
            all_passed = False
            
        return all_passed
        
    except Exception as e:
        print(f"❌ Error importing or testing vector search: {e}")
        return False


def validate_storage_service_imports():
    """Validate that storage services have correct imports and functions."""
    print("\n=== Testing Storage Service Imports ===")
    
    all_passed = True
    
    # Test document storage service
    try:
        from server.services.storage.document_storage_service import get_dimension_column_name
        print("✅ Document storage service imports get_dimension_column_name")
    except ImportError as e:
        print(f"❌ Document storage service missing import: {e}")
        all_passed = False
    except Exception as e:
        print(f"❌ Document storage service import error: {e}")
        all_passed = False
    
    # Test code storage service
    try:
        from server.services.storage.code_storage_service import get_dimension_column_name
        print("✅ Code storage service imports get_dimension_column_name")
    except ImportError as e:
        print(f"❌ Code storage service missing import: {e}")
        all_passed = False
    except Exception as e:
        print(f"❌ Code storage service import error: {e}")
        all_passed = False
        
    return all_passed


def check_file_modifications():
    """Check that the specific lines mentioned in tasks were modified correctly."""
    print("\n=== Checking File Modifications ===")
    
    all_passed = True
    
    # Check vector_search_service.py for build_rpc_params usage
    try:
        with open('/root/Archon-V2-Alpha/python/src/server/services/search/vector_search_service.py', 'r') as f:
            content = f.read()
            
        if 'build_rpc_params(' in content:
            print("✅ vector_search_service.py uses build_rpc_params function")
        else:
            print("❌ vector_search_service.py missing build_rpc_params usage")
            all_passed = False
            
        if 'query_embedding_768' in content:
            print("✅ vector_search_service.py contains dimension-specific parameters")
        else:
            print("❌ vector_search_service.py missing dimension-specific parameters")
            all_passed = False
            
    except Exception as e:
        print(f"❌ Error checking vector_search_service.py: {e}")
        all_passed = False
    
    # Check document_storage_service.py for dynamic column usage
    try:
        with open('/root/Archon-V2-Alpha/python/src/server/services/storage/document_storage_service.py', 'r') as f:
            content = f.read()
            
        if 'get_dimension_column_name(' in content:
            print("✅ document_storage_service.py uses get_dimension_column_name")
        else:
            print("❌ document_storage_service.py missing get_dimension_column_name usage")
            all_passed = False
            
        if 'column_name: batch_embeddings[j]' in content or 'column_name:' in content:
            print("✅ document_storage_service.py uses dynamic column assignment")
        else:
            print("❌ document_storage_service.py missing dynamic column assignment")
            all_passed = False
            
    except Exception as e:
        print(f"❌ Error checking document_storage_service.py: {e}")
        all_passed = False
    
    # Check code_storage_service.py for dynamic column usage
    try:
        with open('/root/Archon-V2-Alpha/python/src/server/services/storage/code_storage_service.py', 'r') as f:
            content = f.read()
            
        if 'get_dimension_column_name(' in content:
            print("✅ code_storage_service.py uses get_dimension_column_name")
        else:
            print("❌ code_storage_service.py missing get_dimension_column_name usage")
            all_passed = False
            
        if 'column_name: embedding' in content:
            print("✅ code_storage_service.py uses dynamic column assignment")
        else:
            print("❌ code_storage_service.py missing dynamic column assignment")
            all_passed = False
            
    except Exception as e:
        print(f"❌ Error checking code_storage_service.py: {e}")
        all_passed = False
        
    return all_passed


def run_integration_validation():
    """Run a simulated integration test."""
    print("\n=== Integration Validation ===")
    
    try:
        from server.services.embeddings.embedding_service import get_dimension_column_name
        from server.services.search.vector_search_service import build_rpc_params
        
        # Simulate full workflow for different dimensions
        test_embeddings = [
            ([0.1] * 768, "text-embedding-3-small (768)"),
            ([0.2] * 1536, "text-embedding-ada-002 (1536)"), 
            ([0.3] * 3072, "text-embedding-3-large (3072)")
        ]
        
        all_passed = True
        print("Simulating embedding storage and search workflow:")
        
        for embedding, model_desc in test_embeddings:
            dims = len(embedding)
            
            # Test storage column selection
            storage_column = get_dimension_column_name(dims)
            print(f"  {model_desc}:")
            print(f"    Storage column: {storage_column}")
            
            # Test search parameter building
            search_params = build_rpc_params(embedding, match_count=5)
            search_param_key = f"query_embedding_{dims}"
            
            if search_param_key in search_params:
                print(f"    Search parameter: {search_param_key} ✅")
            else:
                print(f"    Search parameter: Missing {search_param_key} ❌")
                all_passed = False
            
            # Validate consistency
            expected_column = f"embedding_{dims}"
            if storage_column == expected_column:
                print(f"    Consistency: Storage and search use same dimension ✅")
            else:
                print(f"    Consistency: Mismatch between storage ({storage_column}) and expected ({expected_column}) ❌")
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Integration validation failed: {e}")
        return False


def main():
    """Run all QA validations."""
    print("🔍 Multi-Dimensional Vector System QA Validation")
    print("=" * 60)
    
    tests = [
        ("Dimension Column Mapping", test_dimension_column_mapping),
        ("Vector Search RPC Parameters", test_vector_search_rpc_params),
        ("Storage Service Imports", validate_storage_service_imports),
        ("File Modifications", check_file_modifications),
        ("Integration Validation", run_integration_validation)
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        results[test_name] = test_func()
    
    print("\n" + "=" * 60)
    print("📊 QA VALIDATION SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL" 
        print(f"{test_name:.<40} {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED - Multi-dimensional vector system is working correctly!")
        print("✅ Tasks ready to move from 'review' to 'done' status")
    else:
        print("⚠️  SOME TESTS FAILED - Issues need to be addressed before deployment")
        print("❌ Tasks should remain in 'review' or move back to 'doing' status")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)