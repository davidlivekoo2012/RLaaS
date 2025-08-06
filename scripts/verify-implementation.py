#!/usr/bin/env python3
"""
RLaaS Implementation Verification Script

This script verifies that all core components of the RLaaS platform
are properly implemented and can be imported/initialized.
"""

import sys
import traceback
from typing import Dict, List, Tuple
import importlib
import inspect


class ImplementationVerifier:
    """Verifies RLaaS implementation completeness."""
    
    def __init__(self):
        self.results: Dict[str, Dict[str, bool]] = {}
        self.errors: List[str] = []
    
    def verify_all(self) -> Dict[str, Dict[str, bool]]:
        """Verify all components."""
        
        print("🔍 RLaaS Implementation Verification")
        print("=" * 50)
        
        # Core components
        self.verify_core_components()
        
        # Model management
        self.verify_model_management()
        
        # API components
        self.verify_api_components()
        
        # Configuration
        self.verify_configuration()
        
        # Print summary
        self.print_summary()
        
        return self.results
    
    def verify_core_components(self):
        """Verify core platform components."""
        
        print("\n📦 Core Components")
        print("-" * 30)
        
        core_components = {
            "optimization_engine": "rlaas.core.optimization.engine.OptimizationEngine",
            "nsga3_algorithm": "rlaas.core.optimization.algorithms.NSGAIIIAlgorithm",
            "moead_algorithm": "rlaas.core.optimization.algorithms.MOEADAlgorithm",
            "conflict_resolver": "rlaas.core.optimization.conflict_resolver.TOPSISResolver",
            "pareto_frontier": "rlaas.core.optimization.pareto.ParetoFrontier",
            "policy_engine": "rlaas.core.policy.engine.PolicyEngine",
            "sac_agent": "rlaas.core.policy.agents.SACAgent",
            "ppo_agent": "rlaas.core.policy.agents.PPOAgent",
            "network_environment": "rlaas.core.policy.environments.NetworkEnvironment",
            "recommendation_environment": "rlaas.core.policy.environments.RecommendationEnvironment",
            "adaptive_scheduler": "rlaas.core.scheduler.engine.AdaptiveScheduler",
        }
        
        self.results["core"] = {}
        
        for component_name, import_path in core_components.items():
            try:
                module_path, class_name = import_path.rsplit(".", 1)
                module = importlib.import_module(module_path)
                component_class = getattr(module, class_name)
                
                # Try to instantiate (basic check)
                if component_name in ["optimization_engine", "policy_engine", "adaptive_scheduler"]:
                    instance = component_class()
                    print(f"  ✅ {component_name}: {class_name}")
                else:
                    print(f"  ✅ {component_name}: {class_name}")
                
                self.results["core"][component_name] = True
                
            except Exception as e:
                print(f"  ❌ {component_name}: {str(e)}")
                self.results["core"][component_name] = False
                self.errors.append(f"Core component {component_name}: {str(e)}")
    
    def verify_model_management(self):
        """Verify model management components."""
        
        print("\n🗄️ Model Management")
        print("-" * 30)
        
        model_components = {
            "model_registry": "rlaas.models.registry.ModelRegistry",
            "model_metadata": "rlaas.models.metadata.ModelMetadata",
            "model_info": "rlaas.models.metadata.ModelInfo",
        }
        
        self.results["models"] = {}
        
        for component_name, import_path in model_components.items():
            try:
                module_path, class_name = import_path.rsplit(".", 1)
                module = importlib.import_module(module_path)
                component_class = getattr(module, class_name)
                
                print(f"  ✅ {component_name}: {class_name}")
                self.results["models"][component_name] = True
                
            except Exception as e:
                print(f"  ❌ {component_name}: {str(e)}")
                self.results["models"][component_name] = False
                self.errors.append(f"Model component {component_name}: {str(e)}")
    
    def verify_api_components(self):
        """Verify API components."""
        
        print("\n🌐 API Components")
        print("-" * 30)
        
        api_components = {
            "main_app": "rlaas.core.api.main.app",
            "optimization_routes": "rlaas.core.api.routes.optimization.router",
            "health_routes": "rlaas.core.api.routes.health.router",
            "middleware": "rlaas.core.api.middleware.RequestLoggingMiddleware",
        }
        
        self.results["api"] = {}
        
        for component_name, import_path in api_components.items():
            try:
                module_path, object_name = import_path.rsplit(".", 1)
                module = importlib.import_module(module_path)
                component = getattr(module, object_name)
                
                print(f"  ✅ {component_name}: {object_name}")
                self.results["api"][component_name] = True
                
            except Exception as e:
                print(f"  ❌ {component_name}: {str(e)}")
                self.results["api"][component_name] = False
                self.errors.append(f"API component {component_name}: {str(e)}")
    
    def verify_configuration(self):
        """Verify configuration system."""
        
        print("\n⚙️ Configuration")
        print("-" * 30)
        
        try:
            from rlaas.config import get_config, Config
            
            config = get_config()
            
            # Check key configuration sections
            sections = [
                "database", "redis", "kafka", "mlflow", 
                "optimization", "security", "monitoring"
            ]
            
            self.results["config"] = {}
            
            for section in sections:
                if hasattr(config, section):
                    print(f"  ✅ {section}: configured")
                    self.results["config"][section] = True
                else:
                    print(f"  ❌ {section}: missing")
                    self.results["config"][section] = False
            
            print(f"  ✅ config_class: {type(config).__name__}")
            
        except Exception as e:
            print(f"  ❌ configuration: {str(e)}")
            self.results["config"] = {"error": False}
            self.errors.append(f"Configuration: {str(e)}")
    
    def verify_cli(self):
        """Verify CLI components."""
        
        print("\n💻 CLI Components")
        print("-" * 30)
        
        try:
            from rlaas.cli import cli
            
            print(f"  ✅ cli: {type(cli).__name__}")
            self.results["cli"] = {"main": True}
            
        except Exception as e:
            print(f"  ❌ cli: {str(e)}")
            self.results["cli"] = {"main": False}
            self.errors.append(f"CLI: {str(e)}")
    
    def verify_objectives(self):
        """Verify objective functions."""
        
        print("\n🎯 Objective Functions")
        print("-" * 30)
        
        objective_functions = {
            "latency_objective": "rlaas.core.optimization.objectives.LatencyObjective",
            "throughput_objective": "rlaas.core.optimization.objectives.ThroughputObjective",
            "energy_objective": "rlaas.core.optimization.objectives.EnergyObjective",
            "ctr_objective": "rlaas.core.optimization.objectives.CTRObjective",
            "cvr_objective": "rlaas.core.optimization.objectives.CVRObjective",
            "diversity_objective": "rlaas.core.optimization.objectives.DiversityObjective",
            "5g_problem": "rlaas.core.optimization.objectives.create_5g_optimization_problem",
            "rec_problem": "rlaas.core.optimization.objectives.create_recommendation_optimization_problem",
        }
        
        self.results["objectives"] = {}
        
        for obj_name, import_path in objective_functions.items():
            try:
                module_path, class_or_func_name = import_path.rsplit(".", 1)
                module = importlib.import_module(module_path)
                obj = getattr(module, class_or_func_name)
                
                print(f"  ✅ {obj_name}: {class_or_func_name}")
                self.results["objectives"][obj_name] = True
                
            except Exception as e:
                print(f"  ❌ {obj_name}: {str(e)}")
                self.results["objectives"][obj_name] = False
                self.errors.append(f"Objective {obj_name}: {str(e)}")
    
    def print_summary(self):
        """Print verification summary."""
        
        print("\n📊 Verification Summary")
        print("=" * 50)
        
        total_components = 0
        passed_components = 0
        
        for category, components in self.results.items():
            category_total = len(components)
            category_passed = sum(1 for v in components.values() if v)
            
            total_components += category_total
            passed_components += category_passed
            
            percentage = (category_passed / category_total * 100) if category_total > 0 else 0
            
            status = "✅" if percentage == 100 else "🔄" if percentage >= 50 else "❌"
            print(f"{status} {category.upper()}: {category_passed}/{category_total} ({percentage:.1f}%)")
        
        overall_percentage = (passed_components / total_components * 100) if total_components > 0 else 0
        
        print(f"\n🎯 OVERALL: {passed_components}/{total_components} ({overall_percentage:.1f}%)")
        
        if overall_percentage >= 90:
            print("🎉 Excellent! Implementation is nearly complete.")
        elif overall_percentage >= 75:
            print("👍 Good! Most components are implemented.")
        elif overall_percentage >= 50:
            print("⚠️ Fair. Significant components are missing.")
        else:
            print("❌ Poor. Major implementation work needed.")
        
        if self.errors:
            print(f"\n🐛 Errors Found ({len(self.errors)}):")
            for i, error in enumerate(self.errors[:10], 1):  # Show first 10 errors
                print(f"  {i}. {error}")
            
            if len(self.errors) > 10:
                print(f"  ... and {len(self.errors) - 10} more errors")
    
    def generate_report(self) -> str:
        """Generate a detailed report."""
        
        report = []
        report.append("# RLaaS Implementation Verification Report")
        report.append(f"Generated: {__import__('datetime').datetime.now().isoformat()}")
        report.append("")
        
        for category, components in self.results.items():
            report.append(f"## {category.upper()}")
            report.append("")
            
            for component, status in components.items():
                status_icon = "✅" if status else "❌"
                report.append(f"- {status_icon} {component}")
            
            report.append("")
        
        if self.errors:
            report.append("## Errors")
            report.append("")
            for error in self.errors:
                report.append(f"- {error}")
            report.append("")
        
        return "\n".join(report)


def main():
    """Main verification function."""
    
    verifier = ImplementationVerifier()
    
    # Add CLI verification
    verifier.verify_cli()
    
    # Add objectives verification
    verifier.verify_objectives()
    
    # Run all verifications
    results = verifier.verify_all()
    
    # Generate report
    report = verifier.generate_report()
    
    # Save report
    with open("verification_report.md", "w") as f:
        f.write(report)
    
    print(f"\n📄 Detailed report saved to: verification_report.md")
    
    # Exit with appropriate code
    total_components = sum(len(components) for components in results.values())
    passed_components = sum(
        sum(1 for v in components.values() if v) 
        for components in results.values()
    )
    
    success_rate = (passed_components / total_components) if total_components > 0 else 0
    
    if success_rate >= 0.8:
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Failure


if __name__ == "__main__":
    main()
