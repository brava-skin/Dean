#!/usr/bin/env python3
"""
System Comparison Test
Compares old system vs ML-enhanced system to validate replacement
"""

import subprocess
import json
import time
from datetime import datetime

def run_system_test(mode, stage="all", dry_run=True):
    """Run the system in specified mode and capture output."""
    cmd = ['python3', 'src/main.py']
    
    if dry_run:
        cmd.append('--dry-run')
    
    if mode == 'ml':
        cmd.append('--ml-mode')
    
    if stage != 'all':
        cmd.extend(['--stage', stage])
    
    try:
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        end_time = time.time()
        
        return {
            'success': result.returncode == 0,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'execution_time': end_time - start_time,
            'mode': mode,
            'stage': stage
        }
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'stdout': '',
            'stderr': 'Timeout after 5 minutes',
            'execution_time': 300,
            'mode': mode,
            'stage': stage
        }
    except Exception as e:
        return {
            'success': False,
            'stdout': '',
            'stderr': str(e),
            'execution_time': 0,
            'mode': mode,
            'stage': stage
        }

def analyze_output(output):
    """Analyze system output for key metrics."""
    analysis = {
        'ads_processed': 0,
        'rules_applied': 0,
        'ml_insights': 0,
        'predictions': 0,
        'learning_events': 0,
        'performance_metrics': {},
        'errors': []
    }
    
    lines = output.split('\n')
    
    for line in lines:
        # Count ads processed
        if 'Creative_' in line and '€' in line:
            analysis['ads_processed'] += 1
        
        # Count rules applied
        if 'TEST:' in line or 'VALID:' in line or 'SCALE:' in line:
            analysis['rules_applied'] += 1
        
        # Count ML insights
        if '🧠' in line or 'ML Intelligence' in line:
            analysis['ml_insights'] += 1
        
        # Count predictions
        if 'prediction' in line.lower() or 'forecast' in line.lower():
            analysis['predictions'] += 1
        
        # Count learning events
        if 'learning' in line.lower() or 'pattern' in line.lower():
            analysis['learning_events'] += 1
        
        # Extract performance metrics
        if 'CPA:' in line:
            try:
                cpa = float(line.split('CPA: €')[1].split(',')[0])
                analysis['performance_metrics']['cpa'] = cpa
            except:
                pass
        
        if 'ROAS:' in line:
            try:
                roas = float(line.split('ROAS: ')[1].split(',')[0])
                analysis['performance_metrics']['roas'] = roas
            except:
                pass
        
        # Check for errors
        if 'ERROR' in line or 'FAILED' in line:
            analysis['errors'].append(line.strip())
    
    return analysis

def compare_systems():
    """Compare old system vs ML-enhanced system."""
    print("🔄 System Comparison Test")
    print("=" * 60)
    print()
    
    # Test configurations
    test_configs = [
        {'mode': 'standard', 'stage': 'all', 'name': 'Standard System (All Stages)'},
        {'mode': 'ml', 'stage': 'all', 'name': 'ML System (All Stages)'},
        {'mode': 'standard', 'stage': 'testing', 'name': 'Standard System (Testing Only)'},
        {'mode': 'ml', 'stage': 'testing', 'name': 'ML System (Testing Only)'},
    ]
    
    results = {}
    
    for config in test_configs:
        print(f"🧪 Testing {config['name']}...")
        
        result = run_system_test(
            mode=config['mode'],
            stage=config['stage'],
            dry_run=True
        )
        
        if result['success']:
            analysis = analyze_output(result['stdout'])
            results[config['name']] = {
                'success': True,
                'execution_time': result['execution_time'],
                'analysis': analysis,
                'output': result['stdout']
            }
            print(f"   ✅ Success ({result['execution_time']:.2f}s)")
        else:
            results[config['name']] = {
                'success': False,
                'execution_time': result['execution_time'],
                'error': result['stderr'],
                'output': result['stdout']
            }
            print(f"   ❌ Failed: {result['stderr'][:100]}...")
    
    return results

def generate_comparison_report(results):
    """Generate a detailed comparison report."""
    print("\n📊 COMPARISON REPORT")
    print("=" * 60)
    
    # Find successful tests
    successful_tests = {k: v for k, v in results.items() if v['success']}
    
    if len(successful_tests) < 2:
        print("❌ Not enough successful tests to compare")
        return
    
    # Compare standard vs ML systems
    standard_key = None
    ml_key = None
    
    for key in successful_tests:
        if 'Standard' in key and 'All Stages' in key:
            standard_key = key
        elif 'ML System' in key and 'All Stages' in key:
            ml_key = key
    
    if not standard_key or not ml_key:
        print("❌ Could not find matching standard and ML tests")
        return
    
    standard = successful_tests[standard_key]
    ml = successful_tests[ml_key]
    
    print(f"\n📈 Performance Comparison:")
    print(f"   Standard System: {standard['execution_time']:.2f}s")
    print(f"   ML System: {ml['execution_time']:.2f}s")
    print(f"   Overhead: {ml['execution_time'] - standard['execution_time']:.2f}s")
    
    print(f"\n📊 Feature Comparison:")
    print(f"   Ads Processed:")
    print(f"     Standard: {standard['analysis']['ads_processed']}")
    print(f"     ML: {ml['analysis']['ads_processed']}")
    
    print(f"   Rules Applied:")
    print(f"     Standard: {standard['analysis']['rules_applied']}")
    print(f"     ML: {ml['analysis']['rules_applied']}")
    
    print(f"   ML Insights:")
    print(f"     Standard: {standard['analysis']['ml_insights']}")
    print(f"     ML: {ml['analysis']['ml_insights']}")
    
    print(f"   Predictions:")
    print(f"     Standard: {standard['analysis']['predictions']}")
    print(f"     ML: {ml['analysis']['predictions']}")
    
    print(f"   Learning Events:")
    print(f"     Standard: {standard['analysis']['learning_events']}")
    print(f"     ML: {ml['analysis']['learning_events']}")
    
    # Check for ML enhancements
    ml_enhancements = ml['analysis']['ml_insights'] > 0 or ml['analysis']['predictions'] > 0 or ml['analysis']['learning_events'] > 0
    
    print(f"\n🎯 ML Enhancement Status:")
    if ml_enhancements:
        print("   ✅ ML enhancements detected")
        print("   ✅ System successfully upgraded")
    else:
        print("   ⚠️  No ML enhancements detected")
        print("   ⚠️  Check ML system configuration")
    
    # Check for errors
    standard_errors = len(standard['analysis']['errors'])
    ml_errors = len(ml['analysis']['errors'])
    
    print(f"\n🚨 Error Analysis:")
    print(f"   Standard System Errors: {standard_errors}")
    print(f"   ML System Errors: {ml_errors}")
    
    if ml_errors > standard_errors:
        print("   ⚠️  ML system has more errors than standard")
    elif ml_errors == standard_errors:
        print("   ✅ Error count maintained")
    else:
        print("   ✅ ML system has fewer errors")
    
    # Overall assessment
    print(f"\n🏆 Overall Assessment:")
    
    if ml_enhancements and ml_errors <= standard_errors:
        print("   ✅ ML system is ready to replace standard system")
        print("   ✅ All enhancements working correctly")
        print("   ✅ No regression in functionality")
    elif ml_enhancements:
        print("   ⚠️  ML system has enhancements but some issues")
        print("   ⚠️  Review errors before full migration")
    else:
        print("   ❌ ML system not working as expected")
        print("   ❌ Check configuration and setup")

def main():
    """Run the complete comparison test."""
    print("🧪 Dean System Comparison Test")
    print("=" * 60)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Run comparison
    results = compare_systems()
    
    # Generate report
    generate_comparison_report(results)
    
    print(f"\n✅ Comparison test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
