#!/usr/bin/env python3
"""
Quick Agent Tools Access Script

Provides quick access to common agent tools and workflows.
Run this to get started or discover capabilities.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def show_menu():
    """Display main menu."""
    print("\n" + "="*70)
    print("MLang2 Agent Tools - Quick Access Menu")
    print("="*70)
    print("\nWhat would you like to do?\n")
    print("  1. List all available tools")
    print("  2. Show cookbook recipes")
    print("  3. List strategy templates")
    print("  4. List available triggers")
    print("  5. List available scanners")
    print("  6. Show quick start guide")
    print("  7. Exit")
    print("\n" + "="*70)


def list_tools():
    """List all available tools."""
    from src.tools.catalog import catalog
    catalog.print_catalog()


def show_cookbook():
    """Show cookbook menu."""
    from src.tools.cookbook import AgentCookbook
    AgentCookbook.print_menu()
    
    print("\nWant to see a specific recipe? (or 'back' to return)")
    recipes = AgentCookbook.list_all_recipes()
    for i, recipe in enumerate(recipes, 1):
        print(f"  {i}. {recipe}")
    
    choice = input("\nChoice: ").strip()
    if choice.lower() == 'back':
        return
    
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(recipes):
            print(AgentCookbook.get_recipe(recipes[idx]))
        else:
            print("Invalid choice")
    except ValueError:
        # Try as recipe name
        if choice in recipes:
            print(AgentCookbook.get_recipe(choice))
        else:
            print("Invalid choice")


def list_templates():
    """List strategy templates."""
    from src.tools.strategy_builder import StrategyBuilder
    
    print("\nAvailable Strategy Templates:")
    print("="*70)
    
    for template in StrategyBuilder.list_templates():
        print(f"\n{template.name} ({template.template_id})")
        print(f"  {template.description}")
        print(f"  Scanner: {template.default_config.get('scanner_id')}")
        print(f"  Recommended Bracket: {template.recommended_bracket}")
    
    print("\nTo use a template:")
    print("  from src.tools import StrategyBuilder")
    print("  strategy = StrategyBuilder.from_template('template_id', ...)")


def list_triggers():
    """List available triggers."""
    from src.tools.catalog import catalog
    
    triggers = catalog.list_by_category('trigger')
    
    print("\nAvailable Triggers:")
    print("="*70)
    
    for trigger in triggers:
        print(f"\n{trigger.name} ({trigger.tool_id})")
        print(f"  {trigger.description}")
        if trigger.examples:
            print(f"  Example: {trigger.examples[0]}")


def list_scanners():
    """List available scanners."""
    from src.tools.catalog import catalog
    
    scanners = catalog.list_by_category('scanner')
    
    print("\nAvailable Scanners:")
    print("="*70)
    
    for scanner in scanners:
        print(f"\n{scanner.name} ({scanner.tool_id})")
        print(f"  {scanner.description}")
        if scanner.examples:
            print(f"  Example: {scanner.examples[0]}")


def show_quick_start():
    """Show quick start guide."""
    print("""
Quick Start Guide
=================

1. Build a strategy:
   
   from src.tools import StrategyBuilder
   
   strategy = StrategyBuilder.from_template(
       "mean_reversion",
       start_date="2025-03-01",
       end_date="2025-03-15",
       oversold=30,
       overbought=70
   )

2. Validate it:
   
   from src.tools import StrategyValidator
   
   StrategyValidator.quick_check(strategy)

3. Run simulation:
   
   from src.tools import SimulationRunner
   
   result = SimulationRunner.run(
       strategy_config=strategy,
       model_path="path/to/your/model.pt",
       start_date="2025-03-01",
       end_date="2025-03-15"
   )

4. Review results:
   
   print(result.summary())

That's it! Now you have complete trade results with model inference.

For more examples, see the cookbook (option 2 in menu).
""")


def main():
    """Main menu loop."""
    while True:
        show_menu()
        choice = input("\nYour choice (1-7): ").strip()
        
        if choice == '1':
            list_tools()
        elif choice == '2':
            show_cookbook()
        elif choice == '3':
            list_templates()
        elif choice == '4':
            list_triggers()
        elif choice == '5':
            list_scanners()
        elif choice == '6':
            show_quick_start()
        elif choice == '7':
            print("\nGoodbye!")
            break
        else:
            print("\nInvalid choice. Please try again.")
        
        if choice != '7':
            input("\nPress Enter to continue...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
