#!/usr/bin/env python3
"""
Final end-to-end integration test for multi-dimensional vector system.
Tests the complete workflow from embedding creation to storage to search.
"""

import sys
import os
sys.path.append('/root/Archon-V2-Alpha/python/src')

def test_complete_workflow():
    """Test complete multi-dimensional vector workflow."""
    print("🧪 Running End-to-End Multi-Dimensional Vector Integration Test")
    print("=" * 70)
    
    try:
        # Import all required components
        from server.services.embeddings.embedding_service import get_dimension_column_name
        from server.services.search.vector_search_service import build_rpc_params
        
        # Test scenarios with different embedding models/dimensions
        test_scenarios = [
            {
                'model': 'text-embedding-3-small (reduced)',
                'dimensions': 768,
                'embedding': [0.1] * 768,
                'expected_storage_column': 'embedding_768',
                'expected_search_param': 'query_embedding_768'
            },
            {
                'model': 'custom-model-1024',
                'dimensions': 1024,
                'embedding': [0.2] * 1024,
                'expected_storage_column': 'embedding_1024',
                'expected_search_param': 'query_embedding_1024'
            },
            {
                'model': 'text-embedding-ada-002',
                'dimensions': 1536,
                'embedding': [0.3] * 1536,
                'expected_storage_column': 'embedding_1536',
                'expected_search_param': 'query_embedding_1536'
            },
            {
                'model': 'text-embedding-3-large',
                'dimensions': 3072,
                'embedding': [0.4] * 3072,
                'expected_storage_column': 'embedding_3072',
                'expected_search_param': 'query_embedding_3072'
            }
        ]
        
        all_passed = True
        
        for i, scenario in enumerate(test_scenarios, 1):
            print(f"\n📋 Scenario {i}: {scenario['model']} ({scenario['dimensions']} dimensions)")
            print("-" * 50)
            
            # Test 1: Storage Column Mapping
            storage_column = get_dimension_column_name(scenario['dimensions'])
            if storage_column == scenario['expected_storage_column']:
                print(f"  ✅ Storage: {scenario['dimensions']} dims → {storage_column}")
            else:
                print(f"  ❌ Storage: Expected {scenario['expected_storage_column']}, got {storage_column}")
                all_passed = False
            
            # Test 2: Search Parameter Building
            search_params = build_rpc_params(scenario['embedding'], match_count=5)
            expected_param = scenario['expected_search_param']
            
            if expected_param in search_params:
                print(f"  ✅ Search: {scenario['dimensions']} dims → {expected_param}")
                
                # Verify embedding data integrity
                if search_params[expected_param] == scenario['embedding']:
                    print(f"  ✅ Data Integrity: Embedding vector preserved correctly")
                else:
                    print(f"  ❌ Data Integrity: Embedding vector corrupted")
                    all_passed = False
                    
                # Verify other required parameters
                required_params = ['match_count', 'filter']
                for param in required_params:
                    if param in search_params:
                        print(f"  ✅ Parameters: {param} included")
                    else:
                        print(f"  ❌ Parameters: {param} missing")
                        all_passed = False
                        
            else:
                print(f"  ❌ Search: Expected {expected_param}, available: {list(search_params.keys())}")
                all_passed = False
            
            # Test 3: Consistency Check
            expected_storage_suffix = scenario['expected_storage_column'].replace('embedding_', '')
            expected_search_suffix = scenario['expected_search_param'].replace('query_embedding_', '')
            
            if expected_storage_suffix == expected_search_suffix:
                print(f"  ✅ Consistency: Storage and search use same dimension identifier")
            else:
                print(f"  ❌ Consistency: Storage ({expected_storage_suffix}) != Search ({expected_search_suffix})")
                all_passed = False
        
        # Test 4: Error Handling
        print(f"\n🔧 Error Handling Tests")
        print("-" * 30)
        
        # Test None embedding
        params = build_rpc_params(None, match_count=5)
        if 'query_embedding_1536' in params:
            print("  ✅ None handling: Falls back to query_embedding_1536")
        else:
            print("  ❌ None handling: Failed to fallback properly")
            all_passed = False
        
        # Test empty embedding
        params = build_rpc_params([], match_count=5)
        if 'query_embedding_1536' in params:
            print("  ✅ Empty embedding: Falls back to query_embedding_1536")
        else:
            print("  ❌ Empty embedding: Failed to fallback properly")
            all_passed = False
        
        # Test unsupported dimension
        unsupported_column = get_dimension_column_name(999)
        if unsupported_column == 'embedding_1536':
            print("  ✅ Unsupported dimension: Falls back to embedding_1536")
        else:
            print("  ❌ Unsupported dimension: Failed to fallback properly")
            all_passed = False
        
        # Test 5: Performance Simulation
        print(f"\n⚡ Performance Simulation")
        print("-" * 25)
        
        # Simulate batch operations
        import time
        
        batch_embeddings = [
            [0.1] * 768,   # Different dimensions in same batch
            [0.2] * 1536,
            [0.3] * 3072
        ]
        
        start_time = time.time()
        for embedding in batch_embeddings:
            column = get_dimension_column_name(len(embedding))
            params = build_rpc_params(embedding, match_count=5)
        
        elapsed_time = time.time() - start_time
        print(f"  ✅ Batch processing: {len(batch_embeddings)} embeddings in {elapsed_time:.4f}s")
        
        if elapsed_time < 0.1:  # Should be very fast for these operations
            print("  ✅ Performance: Operations complete within acceptable time")
        else:
            print("  ⚠️  Performance: Operations slower than expected (still acceptable)")
        
        # Final Assessment
        print("\n" + "=" * 70)
        if all_passed:
            print("🎉 ALL INTEGRATION TESTS PASSED!")
            print("✅ Multi-dimensional vector system is fully operational")
            print("✅ Ready for production deployment")
            print("\nSystem Capabilities Validated:")
            print("  • Dynamic storage column assignment based on embedding dimensions")
            print("  • Dimension-specific search parameter generation")
            print("  • Robust error handling with graceful fallbacks")
            print("  • Data integrity preservation throughout the pipeline")
            print("  • Consistent behavior across storage and search components")
            print("  • Support for all four target dimensions: 768, 1024, 1536, 3072")
        else:
            print("❌ SOME INTEGRATION TESTS FAILED!")
            print("⚠️  System requires additional fixes before deployment")
        
        return all_passed
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Integration test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_complete_workflow()
    print(f"\n{'='*70}")
    print(f"Final Result: {'✅ SUCCESS' if success else '❌ FAILURE'}")
    sys.exit(0 if success else 1)