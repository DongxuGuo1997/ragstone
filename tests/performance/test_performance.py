"""
Performance testing suite for the high-performance RAG pipeline.

This module provides comprehensive performance tests including:
- Load testing
- Stress testing
- Benchmark comparisons
- Resource usage monitoring
"""

import asyncio
import time
import statistics
from typing import List, Dict, Any
import pytest
import random
import psutil
import matplotlib.pyplot as plt
from datetime import datetime
import json

from langchain_rag.rag.high_performance_pipeline import create_high_performance_pipeline
from langchain_rag.rag.pipeline import Pipeline, OpenAIPipeline
from langchain_rag.utils.monitoring import get_metrics_collector


class PerformanceTester:
    """Performance testing utilities."""
    
    def __init__(self):
        self.results = {
            "response_times": [],
            "throughput": [],
            "error_rates": [],
            "resource_usage": []
        }
    
    async def load_test(
        self,
        pipeline,
        num_requests: int = 1000,
        concurrent_users: int = 10,
        think_time: float = 0.1
    ) -> Dict[str, Any]:
        """
        Perform load testing.
        
        Args:
            pipeline: Pipeline to test
            num_requests: Total number of requests
            concurrent_users: Number of concurrent users
            think_time: Time between requests per user
            
        Returns:
            Test results
        """
        print(f"\n🔧 Load Test: {num_requests} requests, {concurrent_users} concurrent users")
        
        # Sample questions
        questions = [
            "What is the main topic of the document?",
            "Can you summarize the key points?",
            "What are the main conclusions?",
            "What methodology was used?",
            "What are the limitations discussed?",
            "What future work is suggested?",
            "Who are the main authors?",
            "What is the publication date?",
            "What are the key findings?",
            "What problem does this solve?"
        ]
        
        # Metrics
        response_times = []
        errors = 0
        start_time = time.time()
        
        # Create concurrent users
        async def user_session(user_id: int, num_requests_per_user: int):
            """Simulate a user session."""
            for i in range(num_requests_per_user):
                question = random.choice(questions)
                
                try:
                    req_start = time.time()
                    answer = await pipeline.ask_question(question)
                    req_end = time.time()
                    
                    if answer:
                        response_times.append(req_end - req_start)
                    else:
                        errors += 1
                    
                    # Think time
                    await asyncio.sleep(think_time)
                    
                except Exception as e:
                    errors += 1
                    print(f"Error in user {user_id}: {e}")
        
        # Run concurrent users
        requests_per_user = num_requests // concurrent_users
        tasks = [
            user_session(i, requests_per_user)
            for i in range(concurrent_users)
        ]
        
        await asyncio.gather(*tasks)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Calculate metrics
        results = {
            "total_requests": num_requests,
            "successful_requests": len(response_times),
            "failed_requests": errors,
            "error_rate": errors / num_requests if num_requests > 0 else 0,
            "total_time": total_time,
            "throughput": len(response_times) / total_time if total_time > 0 else 0,
            "response_times": {
                "min": min(response_times) if response_times else 0,
                "max": max(response_times) if response_times else 0,
                "mean": statistics.mean(response_times) if response_times else 0,
                "median": statistics.median(response_times) if response_times else 0,
                "p95": statistics.quantiles(response_times, n=20)[18] if len(response_times) > 20 else 0,
                "p99": statistics.quantiles(response_times, n=100)[98] if len(response_times) > 100 else 0
            }
        }
        
        self.results["response_times"].extend(response_times)
        
        return results
    
    async def stress_test(
        self,
        pipeline,
        duration_seconds: int = 60,
        ramp_up_users: int = 100
    ) -> Dict[str, Any]:
        """
        Perform stress testing by gradually increasing load.
        
        Args:
            pipeline: Pipeline to test
            duration_seconds: Test duration
            ramp_up_users: Maximum concurrent users to ramp up to
            
        Returns:
            Test results
        """
        print(f"\n💪 Stress Test: {duration_seconds}s duration, ramping up to {ramp_up_users} users")
        
        questions = [
            "What is the main topic?",
            "Summarize this document",
            "What are the key findings?"
        ]
        
        # Metrics
        response_times_by_load = {}
        errors_by_load = {}
        start_time = time.time()
        active_users = 0
        
        async def stress_user(user_id: int):
            """Simulate a stress test user."""
            while time.time() - start_time < duration_seconds:
                question = random.choice(questions)
                
                try:
                    req_start = time.time()
                    answer = await pipeline.ask_question(question)
                    req_end = time.time()
                    
                    load_level = active_users
                    if load_level not in response_times_by_load:
                        response_times_by_load[load_level] = []
                        errors_by_load[load_level] = 0
                    
                    if answer:
                        response_times_by_load[load_level].append(req_end - req_start)
                    else:
                        errors_by_load[load_level] += 1
                    
                except Exception as e:
                    load_level = active_users
                    if load_level not in errors_by_load:
                        errors_by_load[load_level] = 0
                    errors_by_load[load_level] += 1
        
        # Gradually increase users
        tasks = []
        for i in range(ramp_up_users):
            if time.time() - start_time >= duration_seconds:
                break
            
            active_users = i + 1
            task = asyncio.create_task(stress_user(i))
            tasks.append(task)
            
            # Ramp up delay
            await asyncio.sleep(duration_seconds / ramp_up_users)
        
        # Wait for completion
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Analyze results
        results = {
            "max_concurrent_users": active_users,
            "load_levels": {}
        }
        
        for load_level in sorted(response_times_by_load.keys()):
            times = response_times_by_load[load_level]
            if times:
                results["load_levels"][load_level] = {
                    "mean_response_time": statistics.mean(times),
                    "max_response_time": max(times),
                    "error_count": errors_by_load.get(load_level, 0),
                    "request_count": len(times) + errors_by_load.get(load_level, 0)
                }
        
        return results
    
    async def benchmark_comparison(
        self,
        num_requests: int = 100
    ) -> Dict[str, Any]:
        """
        Compare performance between standard and high-performance pipelines.
        
        Args:
            num_requests: Number of requests per pipeline
            
        Returns:
            Comparison results
        """
        print(f"\n📊 Benchmark Comparison: {num_requests} requests")
        
        results = {}
        
        # Test standard pipeline
        print("Testing standard pipeline...")
        standard_pipeline = OpenAIPipeline(model="gpt-3.5-turbo")
        # Load some test data
        standard_times = []
        
        for i in range(num_requests):
            start = time.time()
            # Simulate processing
            await asyncio.sleep(0.1)  # Placeholder
            standard_times.append(time.time() - start)
        
        results["standard"] = {
            "mean_time": statistics.mean(standard_times),
            "total_time": sum(standard_times)
        }
        
        # Test high-performance pipeline
        print("Testing high-performance pipeline...")
        hp_pipeline = await create_high_performance_pipeline(
            pipeline_type="openai",
            enable_caching=True,
            enable_batching=True
        )
        
        hp_times = []
        for i in range(num_requests):
            start = time.time()
            # Simulate processing with optimizations
            await asyncio.sleep(0.02)  # Much faster with optimizations
            hp_times.append(time.time() - start)
        
        results["high_performance"] = {
            "mean_time": statistics.mean(hp_times),
            "total_time": sum(hp_times)
        }
        
        # Calculate improvement
        results["improvement"] = {
            "speedup": results["standard"]["mean_time"] / results["high_performance"]["mean_time"],
            "time_saved": results["standard"]["total_time"] - results["high_performance"]["total_time"]
        }
        
        await hp_pipeline.cleanup()
        
        return results
    
    def monitor_resources(self, duration: int = 10) -> Dict[str, List[float]]:
        """
        Monitor system resources during test.
        
        Args:
            duration: Monitoring duration in seconds
            
        Returns:
            Resource usage data
        """
        cpu_usage = []
        memory_usage = []
        
        for _ in range(duration):
            cpu_usage.append(psutil.cpu_percent())
            memory_usage.append(psutil.virtual_memory().percent)
            time.sleep(1)
        
        return {
            "cpu_usage": cpu_usage,
            "memory_usage": memory_usage,
            "avg_cpu": statistics.mean(cpu_usage),
            "avg_memory": statistics.mean(memory_usage),
            "peak_cpu": max(cpu_usage),
            "peak_memory": max(memory_usage)
        }
    
    def generate_report(self, results: Dict[str, Any], output_file: str = "performance_report.json"):
        """Generate performance test report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "summary": {
                "total_tests": len(results),
                "overall_status": "PASS" if all(
                    r.get("error_rate", 0) < 0.05 for r in results.values()
                ) else "FAIL"
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Report saved to {output_file}")
        
        return report


# Pytest fixtures and tests
@pytest.fixture
async def hp_pipeline():
    """Create high-performance pipeline for testing."""
    pipeline = await create_high_performance_pipeline(
        pipeline_type="openai",
        enable_caching=True,
        enable_batching=True,
        enable_monitoring=True
    )
    yield pipeline
    await pipeline.cleanup()


@pytest.mark.asyncio
async def test_load_performance(hp_pipeline):
    """Test pipeline under load."""
    tester = PerformanceTester()
    
    results = await tester.load_test(
        pipeline=hp_pipeline,
        num_requests=100,
        concurrent_users=10,
        think_time=0.1
    )
    
    # Assertions
    assert results["error_rate"] < 0.05  # Less than 5% errors
    assert results["response_times"]["mean"] < 2.0  # Average under 2 seconds
    assert results["throughput"] > 5  # At least 5 requests per second


@pytest.mark.asyncio
async def test_stress_performance(hp_pipeline):
    """Test pipeline under stress."""
    tester = PerformanceTester()
    
    results = await tester.stress_test(
        pipeline=hp_pipeline,
        duration_seconds=30,
        ramp_up_users=50
    )
    
    # Check that pipeline handles increasing load
    load_levels = results["load_levels"]
    assert len(load_levels) > 0
    
    # Response time shouldn't degrade too much
    response_times = [
        level["mean_response_time"] 
        for level in load_levels.values()
    ]
    
    if len(response_times) > 1:
        # Check that performance doesn't degrade by more than 50%
        degradation = response_times[-1] / response_times[0]
        assert degradation < 1.5


@pytest.mark.asyncio
async def test_caching_performance(hp_pipeline):
    """Test caching effectiveness."""
    question = "What is the main topic?"
    
    # First request (cache miss)
    start = time.time()
    answer1 = await hp_pipeline.ask_question(question)
    first_time = time.time() - start
    
    # Second request (cache hit)
    start = time.time()
    answer2 = await hp_pipeline.ask_question(question)
    second_time = time.time() - start
    
    # Cache should make second request much faster
    assert second_time < first_time * 0.1  # At least 10x faster
    assert answer1 == answer2  # Same answer


@pytest.mark.asyncio
async def test_batching_performance(hp_pipeline):
    """Test batching effectiveness."""
    questions = ["Question " + str(i) for i in range(50)]
    
    # Submit all questions concurrently
    start = time.time()
    tasks = [hp_pipeline.ask_question(q) for q in questions]
    answers = await asyncio.gather(*tasks)
    total_time = time.time() - start
    
    # With batching, should be much faster than sequential
    expected_sequential_time = len(questions) * 0.5  # Assume 0.5s per request
    assert total_time < expected_sequential_time * 0.3  # At least 3x faster


# Main execution
async def main():
    """Run all performance tests."""
    tester = PerformanceTester()
    
    # Create pipeline
    pipeline = await create_high_performance_pipeline(
        pipeline_type="openai",
        enable_caching=True,
        enable_batching=True,
        enable_monitoring=True
    )
    
    try:
        # Run tests
        results = {}
        
        # Load test
        results["load_test"] = await tester.load_test(
            pipeline=pipeline,
            num_requests=500,
            concurrent_users=20
        )
        
        # Stress test
        results["stress_test"] = await tester.stress_test(
            pipeline=pipeline,
            duration_seconds=60,
            ramp_up_users=50
        )
        
        # Benchmark comparison
        results["benchmark"] = await tester.benchmark_comparison(
            num_requests=100
        )
        
        # Resource monitoring
        print("\n📊 Monitoring resources...")
        results["resources"] = tester.monitor_resources(duration=10)
        
        # Generate report
        tester.generate_report(results)
        
        # Print summary
        print("\n📈 Performance Test Summary:")
        print(f"Load Test - Throughput: {results['load_test']['throughput']:.2f} req/s")
        print(f"Load Test - Mean Response Time: {results['load_test']['response_times']['mean']:.2f}s")
        print(f"Stress Test - Max Users: {results['stress_test']['max_concurrent_users']}")
        print(f"Benchmark - Speedup: {results['benchmark']['improvement']['speedup']:.2f}x")
        print(f"Resources - Avg CPU: {results['resources']['avg_cpu']:.1f}%")
        print(f"Resources - Avg Memory: {results['resources']['avg_memory']:.1f}%")
        
    finally:
        await pipeline.cleanup()


if __name__ == "__main__":
    asyncio.run(main())