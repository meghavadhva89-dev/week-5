#!/usr/bin/env python3
"""
Comprehensive test script for apputil.py functions
"""

def test_survival_demographics():
    """Test survival_demographics function"""
    print("1. Testing survival_demographics()...")
    from apputil import survival_demographics
    
    result = survival_demographics()
    
    # Test 1: Total number of groups (should be 24: 3 classes × 2 sexes × 4 age groups)
    print(f"   ✓ Total groups: {len(result)} (expected: 24)")
    assert len(result) == 24, f"Expected 24 groups, got {len(result)}"
    
    # Test 2: Age group dtype should be categorical
    age_dtype = str(result['age_group'].dtype)
    print(f"   ✓ Age group dtype: {age_dtype}")
    assert age_dtype == 'category', f"Expected category dtype, got {age_dtype}"
    
    # Test 3: Check categories
    categories = list(result['age_group'].cat.categories)
    expected_categories = ['Child', 'Teen', 'Adult', 'Senior']
    print(f"   ✓ Categories: {categories}")
    assert categories == expected_categories, f"Expected {expected_categories}, got {categories}"
    
    # Test 4: Check for the missing group (second class, female, seniors)
    missing_group = result[
        (result['pclass'] == 2) & 
        (result['sex'] == 'female') & 
        (result['age_group'] == 'Senior')
    ]
    print(f"   ✓ Missing group found: {len(missing_group)} row(s)")
    assert len(missing_group) == 1, f"Expected 1 row for missing group, got {len(missing_group)}"
    
    # Test 5: Check values for missing group
    if len(missing_group) > 0:
        row = missing_group.iloc[0]
        n_pass = row['n_passengers']
        surv_rate = row['survival_rate']
        print(f"   ✓ Missing group values: n_passengers={n_pass}, survival_rate={surv_rate}")
        assert n_pass == 0, f"Expected 0 passengers, got {n_pass}"
        assert surv_rate == 0.0, f"Expected 0.0 survival rate, got {surv_rate}"
    
    # Test 6: Check required columns
    expected_columns = ['pclass', 'sex', 'age_group', 'n_passengers', 'n_survivors', 'survival_rate']
    for col in expected_columns:
        assert col in result.columns, f"Missing column: {col}"
    print(f"   ✓ All required columns present: {expected_columns}")
    
    print("   ✅ survival_demographics() - ALL TESTS PASSED!")
    return True


def test_family_groups():
    """Test family_groups function"""
    print("\n2. Testing family_groups()...")
    from apputil import family_groups
    
    result = family_groups()
    
    # Test that we have data
    expected_columns = ['family_size', 'pclass', 'n_passengers', 'avg_fare', 'min_fare', 'max_fare']
    for col in expected_columns:
        assert col in result.columns, f"Missing column: {col}"
    print(f"   ✓ All required columns present: {expected_columns}")
    
    # Test that we have data
    assert len(result) > 0, "No data returned"
    print(f"   ✓ Data returned: {len(result)} groups")
    
    print("   ✅ family_groups() - ALL TESTS PASSED!")
    return True


def test_last_names():
    """Test last_names function"""
    print("\n3. Testing last_names()...")
    from apputil import last_names
    
    result = last_names()
    
    # Test that it returns a Series
    import pandas as pd
    assert isinstance(result, pd.Series), f"Expected Series, got {type(result)}"
    print(f"   ✓ Returns pandas Series with {len(result)} unique names")
    
    # Test that it has some common names
    assert len(result) > 0, "No names returned"
    print(f"   ✓ Top 3 most common names: {result.head(3).to_dict()}")
    
    print("   ✅ last_names() - ALL TESTS PASSED!")
    return True


def test_determine_age_division():
    """Test determine_age_division function"""
    print("\n4. Testing determine_age_division()...")
    from apputil import determine_age_division
    
    result = determine_age_division()
    
    # Test that older_passenger column exists
    assert 'older_passenger' in result.columns, "Missing older_passenger column"
    print("   ✓ older_passenger column exists")
    
    # Test that it's boolean type
    dtype_str = str(result['older_passenger'].dtype)
    assert dtype_str == 'bool', f"Expected bool dtype, got {dtype_str}"
    print(f"   ✓ older_passenger dtype: {dtype_str}")
    
    # Test that we have both True and False values
    true_count = result['older_passenger'].sum()
    false_count = (~result['older_passenger']).sum()
    total_count = len(result)
    print(f"   ✓ True values: {true_count}, False values: {false_count}, Total: {total_count}")
    
    # Test no NaN values
    nan_count = result['older_passenger'].isna().sum()
    assert nan_count == 0, f"Expected 0 NaN values, got {nan_count}"
    print(f"   ✓ NaN values: {nan_count}")
    
    print("   ✅ determine_age_division() - ALL TESTS PASSED!")
    return True


def test_visualization_functions():
    """Test that visualization functions work"""
    print("\n5. Testing visualization functions...")
    from apputil import visualize_demographic, visualize_families, visualize_age_division
    
    # Test that they return plotly figures
    fig1 = visualize_demographic()
    fig2 = visualize_families()
    fig3 = visualize_age_division()
    
    # Basic check that they return something (plotly figures)
    assert fig1 is not None, "visualize_demographic returned None"
    assert fig2 is not None, "visualize_families returned None"
    assert fig3 is not None, "visualize_age_division returned None"
    
    print("   ✓ visualize_demographic() - OK")
    print("   ✓ visualize_families() - OK")
    print("   ✓ visualize_age_division() - OK")
    print("   ✅ All visualization functions - TESTS PASSED!")
    return True


def main():
    """Run all tests"""
    print("🧪 COMPREHENSIVE TESTING OF APPUTIL.PY FUNCTIONS")
    print("=" * 55)
    
    try:
        # Run all tests
        test_survival_demographics()
        test_family_groups() 
        test_last_names()
        test_determine_age_division()
        test_visualization_functions()
        
        print("\n" + "=" * 55)
        print("🎉 ALL TESTS PASSED! Your functions are working correctly!")
        print("=" * 55)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise


if __name__ == "__main__":
    main()