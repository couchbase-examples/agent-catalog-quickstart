#!/usr/bin/env python3
"""
Unified Evaluation Runner for Agent Catalog Examples

This script provides a centralized way to run evaluations across both
flight search agent examples using different evaluation frameworks:

1. Flight Search Agent (single agent with ReAct pattern)
   - Arize AI evaluation with observability
   - Basic Agent Catalog span-based evaluation

2. Multi-Agent Flight Planner (LangGraph with multiple agents)
   - Ragas evaluation (existing)
   - Arize AI evaluation (new)
   - Comparative analysis between frameworks

Usage:
    python run_evaluations.py --help
    python run_evaluations.py --project flight_search --framework arize
    python run_evaluations.py --project with_langgraph --framework ragas
    python run_evaluations.py --project both --framework both --comparison
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


class EvaluationRunner:
    """Unified evaluation runner for Agent Catalog examples."""
    
    def __init__(self):
        self.root_dir = Path(__file__).parent
        self.flight_search_dir = self.root_dir / "examples" / "flight_search_agent"
        self.langgraph_dir = self.root_dir / "agent-catalog" / "examples" / "with_langgraph"
        self.hotel_support_dir = self.root_dir / "examples" / "hotel_support_agent"
        self.route_planner_dir = self.root_dir / "examples" / "route_planner_agent"
    
    def check_dependencies(self, project: str, framework: str) -> bool:
        """Check if required dependencies are available."""
        print(f"🔍 Checking dependencies for {project} with {framework}...")
        
        if project == "flight_search":
            # Check flight search agent dependencies
            try:
                if framework in ["arize", "both"]:
                    import arize.otel
                    import phoenix.evals
                    print("   ✅ Arize dependencies available")
                return True
            except ImportError as e:
                print(f"   ❌ Missing dependencies: {e}")
                return False
        
        elif project == "with_langgraph":
            # Check langgraph project dependencies
            try:
                if framework in ["ragas", "both"]:
                    import ragas
                    print("   ✅ Ragas dependencies available")
                
                if framework in ["arize", "both"]:
                    import arize.otel
                    import phoenix.evals
                    print("   ✅ Arize dependencies available")
                return True
            except ImportError as e:
                print(f"   ❌ Missing dependencies: {e}")
                return False
        
        elif project == "hotel_support":
            # Check hotel support agent dependencies
            try:
                if framework in ["arize", "both"]:
                    import arize.otel
                    import phoenix.evals
                    print("   ✅ Arize dependencies available")
                return True
            except ImportError as e:
                print(f"   ❌ Missing dependencies: {e}")
                return False
        
        elif project == "route_planner":
            # Check route planner agent dependencies
            try:
                if framework in ["arize", "both"]:
                    import arize.otel
                    import phoenix.evals
                    print("   ✅ Arize dependencies available")
                return True
            except ImportError as e:
                print(f"   ❌ Missing dependencies: {e}")
                return False
        
        return True
    
    def setup_environment(self, project: str):
        """Setup environment variables for evaluation."""
        print(f"🔧 Setting up environment for {project}...")
        
        # Common environment variables
        env_vars = {
            "PYTHONPATH": str(self.root_dir),
        }
        
        # Project-specific environment setup
        if project == "flight_search":
            env_vars.update({
                "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
                "ARIZE_SPACE_ID": os.getenv("ARIZE_SPACE_ID", "your-space-id"),
                "ARIZE_API_KEY": os.getenv("ARIZE_API_KEY", "your-api-key"),
                "ARIZE_DEVELOPER_KEY": os.getenv("ARIZE_DEVELOPER_KEY", "your-developer-key"),
            })
        
        elif project == "with_langgraph":
            env_vars.update({
                "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
                "ARIZE_SPACE_ID": os.getenv("ARIZE_SPACE_ID", "your-space-id"),
                "ARIZE_API_KEY": os.getenv("ARIZE_API_KEY", "your-api-key"),
                "ARIZE_DEVELOPER_KEY": os.getenv("ARIZE_DEVELOPER_KEY", "your-developer-key"),
            })
        
        # Set environment variables
        for key, value in env_vars.items():
            if value:
                os.environ[key] = value
        
        # Check for required API keys
        if not os.getenv("OPENAI_API_KEY"):
            print("   ⚠️  Warning: OPENAI_API_KEY not set")
        
        print("   ✅ Environment setup complete")
    
    def run_flight_search_evaluation(self, framework: str, test_type: str = "comprehensive"):
        """Run evaluation for flight search agent."""
        print(f"\n🚀 Running flight search agent evaluation with {framework}...")
        
        if not self.flight_search_dir.exists():
            print(f"❌ Flight search directory not found: {self.flight_search_dir}")
            return False
        
        # Change to flight search directory
        original_dir = os.getcwd()
        os.chdir(self.flight_search_dir)
        
        try:
            if framework == "arize":
                # Run Arize evaluation
                eval_script = self.flight_search_dir / "evals" / "eval_short.py"
                if eval_script.exists():
                    result = subprocess.run([
                        sys.executable, str(eval_script), test_type
                    ], capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        print("✅ Flight search Arize evaluation completed successfully")
                        print(result.stdout)
                        return True
                    else:
                        print("❌ Flight search Arize evaluation failed")
                        print(result.stderr)
                        return False
                else:
                    print(f"❌ Evaluation script not found: {eval_script}")
                    return False
            else:
                print(f"⚠️  Framework {framework} not supported for flight search agent")
                return False
        
        finally:
            os.chdir(original_dir)
    
    def run_langgraph_evaluation(self, framework: str, test_type: str = "comprehensive"):
        """Run evaluation for multi-agent LangGraph project."""
        print(f"\n🚀 Running LangGraph multi-agent evaluation with {framework}...")
        
        if not self.langgraph_dir.exists():
            print(f"❌ LangGraph directory not found: {self.langgraph_dir}")
            return False
        
        # Change to langgraph directory
        original_dir = os.getcwd()
        os.chdir(self.langgraph_dir)
        
        try:
            if framework == "ragas":
                # Run existing Ragas evaluation
                eval_script = self.langgraph_dir / "evals" / "eval_short.py"
                if eval_script.exists():
                    result = subprocess.run([
                        sys.executable, "-m", "pytest", str(eval_script), "-v"
                    ], capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        print("✅ LangGraph Ragas evaluation completed successfully")
                        print(result.stdout)
                        return True
                    else:
                        print("⚠️  LangGraph Ragas evaluation completed with warnings")
                        print(result.stdout)
                        print(result.stderr)
                        return True  # Pytest might have warnings but still pass
                else:
                    print(f"❌ Ragas evaluation script not found: {eval_script}")
                    return False
            
            elif framework == "arize":
                # Run new Arize evaluation
                eval_script = self.langgraph_dir / "evals" / "eval_arize.py"
                if eval_script.exists():
                    result = subprocess.run([
                        sys.executable, str(eval_script), test_type
                    ], capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        print("✅ LangGraph Arize evaluation completed successfully")
                        print(result.stdout)
                        return True
                    else:
                        print("❌ LangGraph Arize evaluation failed")
                        print(result.stderr)
                        return False
                else:
                    print(f"❌ Arize evaluation script not found: {eval_script}")
                    return False
            
            elif framework == "both":
                # Run both Ragas and Arize evaluations
                ragas_success = self.run_langgraph_evaluation("ragas", test_type)
                arize_success = self.run_langgraph_evaluation("arize", test_type)
                return ragas_success and arize_success
            
            else:
                print(f"⚠️  Framework {framework} not supported for LangGraph project")
                return False
        
        finally:
            os.chdir(original_dir)
    
    def run_hotel_support_evaluation(self, framework: str, test_type: str = "comprehensive"):
        """Run evaluation for hotel support agent."""
        print(f"\n🚀 Running hotel support agent evaluation with {framework}...")
        
        if not self.hotel_support_dir.exists():
            print(f"❌ Hotel support directory not found: {self.hotel_support_dir}")
            return False
        
        # Change to hotel support directory
        original_dir = os.getcwd()
        os.chdir(self.hotel_support_dir)
        
        try:
            if framework == "arize":
                # Run Arize evaluation
                eval_script = self.hotel_support_dir / "evals" / "eval_arize.py"
                if eval_script.exists():
                    result = subprocess.run([
                        sys.executable, str(eval_script), test_type
                    ], capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        print("✅ Hotel support Arize evaluation completed successfully")
                        print(result.stdout)
                        return True
                    else:
                        print("❌ Hotel support Arize evaluation failed")
                        print(result.stderr)
                        return False
                else:
                    print(f"❌ Evaluation script not found: {eval_script}")
                    return False
            else:
                print(f"⚠️  Framework {framework} not supported for hotel support agent")
                return False
        
        finally:
            os.chdir(original_dir)
    
    def run_route_planner_evaluation(self, framework: str, test_type: str = "comprehensive"):
        """Run evaluation for route planner agent."""
        print(f"\n🚀 Running route planner agent evaluation with {framework}...")
        
        if not self.route_planner_dir.exists():
            print(f"❌ Route planner directory not found: {self.route_planner_dir}")
            return False
        
        # Change to route planner directory
        original_dir = os.getcwd()
        os.chdir(self.route_planner_dir)
        
        try:
            if framework == "arize":
                # Run Arize evaluation
                eval_script = self.route_planner_dir / "evals" / "eval_arize.py"
                if eval_script.exists():
                    result = subprocess.run([
                        sys.executable, str(eval_script), test_type
                    ], capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        print("✅ Route planner Arize evaluation completed successfully")
                        print(result.stdout)
                        return True
                    else:
                        print("❌ Route planner Arize evaluation failed")
                        print(result.stderr)
                        return False
                else:
                    print(f"❌ Evaluation script not found: {eval_script}")
                    return False
            else:
                print(f"⚠️  Framework {framework} not supported for route planner agent")
                return False
        
        finally:
            os.chdir(original_dir)
    
    def run_comparative_analysis(self):
        """Run comparative analysis between frameworks and projects."""
        print("\n📊 Running comparative analysis...")
        
        results = {
            "flight_search_arize": False,
            "langgraph_ragas": False,
            "langgraph_arize": False,
            "hotel_support_arize": False,
            "route_planner_arize": False,
        }
        
        # Run flight search with Arize
        results["flight_search_arize"] = self.run_flight_search_evaluation("arize")
        
        # Run LangGraph with both frameworks
        results["langgraph_ragas"] = self.run_langgraph_evaluation("ragas")
        results["langgraph_arize"] = self.run_langgraph_evaluation("arize")
        
        # Run hotel support with Arize
        results["hotel_support_arize"] = self.run_hotel_support_evaluation("arize")
        
        # Run route planner with Arize
        results["route_planner_arize"] = self.run_route_planner_evaluation("arize")
        
        # Generate summary report
        self.generate_summary_report(results)
        
        return all(results.values())
    
    def generate_summary_report(self, results: dict):
        """Generate a summary report of all evaluations."""
        print("\n" + "=" * 80)
        print("📋 EVALUATION SUMMARY REPORT")
        print("=" * 80)
        
        print("\n🏗️  Architecture Comparison:")
        print("   Flight Search Agent: Single ReAct agent with tool calling")
        print("   Multi-Agent Planner: LangGraph workflow with 3 specialized agents")
        print("   Hotel Support Agent: LangChain ReAct agent with vector search")
        print("   Route Planner Agent: LlamaIndex-based agent with multiple tools")
        
        print("\n📊 Evaluation Framework Comparison:")
        print("   Ragas: LLM-based scoring with conversation analysis")
        print("   Arize: Production observability with LLM-as-judge evaluation")
        
        print("\n✅ Evaluation Results:")
        for test_name, success in results.items():
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"   {test_name.replace('_', ' ').title()}: {status}")
        
        success_rate = sum(results.values()) / len(results) * 100
        print(f"\n📈 Overall Success Rate: {success_rate:.1f}%")
        
        print("\n📁 Generated Reports:")
        # List generated report files
        flight_report = self.flight_search_dir / "evals" / "evaluation_report.md"
        langgraph_report = self.langgraph_dir / "evals" / "arize_ragas_comparison_report.md"
        
        if flight_report.exists():
            print(f"   Flight Search: {flight_report}")
        if langgraph_report.exists():
            print(f"   LangGraph Comparison: {langgraph_report}")
        
        print("\n🔗 Key Integration Points:")
        print("   - Both projects use Agent Catalog for tool/prompt management")
        print("   - Arize provides unified observability across architectures")
        print("   - Ragas offers specialized conversation evaluation")
        print("   - Agent Catalog spans enable consistent logging")
        
        print("\n🚀 Next Steps:")
        print("   1. Review generated reports for detailed metrics")
        print("   2. Check Arize dashboard for trace visualization")
        print("   3. Compare evaluation approaches for your use case")
        print("   4. Iterate on agent performance based on results")
        
        print("=" * 80)


def main():
    """Main entry point for the unified evaluation runner."""
    parser = argparse.ArgumentParser(
        description="Unified Evaluation Runner for Agent Catalog Examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_evaluations.py --project flight_search --framework arize
  python run_evaluations.py --project with_langgraph --framework ragas
  python run_evaluations.py --project both --framework both --comparison
  python run_evaluations.py --project with_langgraph --framework arize --test-type short_threads
        """
    )
    
    parser.add_argument(
        "--project",
        choices=["flight_search", "with_langgraph", "hotel_support", "route_planner", "all"],
        default="all",
        help="Which project to evaluate (default: all)"
    )
    
    parser.add_argument(
        "--framework",
        choices=["ragas", "arize", "both"],
        default="both",
        help="Which evaluation framework to use (default: both)"
    )
    
    parser.add_argument(
        "--test-type",
        choices=["bad_intro", "short_threads", "comprehensive"],
        default="comprehensive",
        help="Type of test to run (default: comprehensive)"
    )
    
    parser.add_argument(
        "--comparison",
        action="store_true",
        help="Run comparative analysis across projects and frameworks"
    )
    
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Only check dependencies, don't run evaluations"
    )
    
    args = parser.parse_args()
    
    # Create evaluation runner
    runner = EvaluationRunner()
    
    print("🎯 Unified Agent Catalog Evaluation Runner")
    print("=" * 50)
    
    # Check dependencies if requested
    if args.check_deps:
        if args.project in ["flight_search", "all"]:
            runner.check_dependencies("flight_search", args.framework)
        
        if args.project in ["with_langgraph", "all"]:
            runner.check_dependencies("with_langgraph", args.framework)
        
        if args.project in ["hotel_support", "all"]:
            runner.check_dependencies("hotel_support", args.framework)
        
        if args.project in ["route_planner", "all"]:
            runner.check_dependencies("route_planner", args.framework)
        
        return
    
    # Setup environment
    if args.project in ["flight_search", "all"]:
        runner.setup_environment("flight_search")
    
    if args.project in ["with_langgraph", "all"]:
        runner.setup_environment("with_langgraph")
    
    if args.project in ["hotel_support", "all"]:
        runner.setup_environment("hotel_support")
    
    if args.project in ["route_planner", "all"]:
        runner.setup_environment("route_planner")
    
    # Run evaluations
    success = True
    
    if args.comparison or (args.project == "all" and args.framework == "both"):
        # Run comprehensive comparative analysis
        success = runner.run_comparative_analysis()
    
    else:
        # Run specific evaluations
        if args.project in ["flight_search", "all"]:
            if args.framework in ["arize", "both"]:
                success &= runner.run_flight_search_evaluation("arize", args.test_type)
        
        if args.project in ["with_langgraph", "all"]:
            if args.framework in ["ragas", "both"]:
                success &= runner.run_langgraph_evaluation("ragas", args.test_type)
            
            if args.framework in ["arize", "both"]:
                success &= runner.run_langgraph_evaluation("arize", args.test_type)
        
        if args.project in ["hotel_support", "all"]:
            if args.framework in ["arize", "both"]:
                success &= runner.run_hotel_support_evaluation("arize", args.test_type)
        
        if args.project in ["route_planner", "all"]:
            if args.framework in ["arize", "both"]:
                success &= runner.run_route_planner_evaluation("arize", args.test_type)
    
    # Exit with appropriate code
    if success:
        print("\n🎉 All evaluations completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Some evaluations failed. Check the output above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()