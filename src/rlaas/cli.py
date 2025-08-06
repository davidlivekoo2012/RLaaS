"""
Command Line Interface for RLaaS platform.
"""

import click
import asyncio
import json
import sys
from typing import Dict, Any
import httpx
from rlaas.config import get_config

config = get_config()


@click.group()
@click.option('--api-url', default='http://localhost:8000', help='RLaaS API URL')
@click.pass_context
def cli(ctx, api_url):
    """RLaaS Command Line Interface."""
    ctx.ensure_object(dict)
    ctx.obj['api_url'] = api_url


@cli.group()
def optimize():
    """Multi-objective optimization commands."""
    pass


@optimize.command()
@click.option('--problem-type', required=True, type=click.Choice(['5g', 'recommendation']), 
              help='Type of optimization problem')
@click.option('--algorithm', default='nsga3', type=click.Choice(['nsga3', 'moead']),
              help='Optimization algorithm')
@click.option('--mode', default='normal', 
              type=click.Choice(['normal', 'emergency', 'revenue_focused', 'user_experience']),
              help='Optimization mode')
@click.option('--population-size', default=100, help='Population size')
@click.option('--generations', default=500, help='Number of generations')
@click.option('--weights', help='Objective weights as JSON string')
@click.option('--timeout', help='Timeout in seconds')
@click.pass_context
def start(ctx, problem_type, algorithm, mode, population_size, generations, weights, timeout):
    """Start a new optimization."""
    
    api_url = ctx.obj['api_url']
    
    # Prepare request data
    data = {
        'problem_type': problem_type,
        'algorithm': algorithm,
        'mode': mode,
        'population_size': population_size,
        'generations': generations
    }
    
    if weights:
        try:
            data['weights'] = json.loads(weights)
        except json.JSONDecodeError:
            click.echo("Error: Invalid JSON format for weights", err=True)
            sys.exit(1)
    
    if timeout:
        data['timeout'] = int(timeout)
    
    # Make API request
    async def make_request():
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(f"{api_url}/api/v1/optimization/optimize", json=data)
                response.raise_for_status()
                result = response.json()
                
                click.echo(f"Optimization started successfully!")
                click.echo(f"Optimization ID: {result['optimization_id']}")
                click.echo(f"Status: {result['status']}")
                click.echo(f"Use 'rlaas optimize status {result['optimization_id']}' to check progress")
                
            except httpx.HTTPError as e:
                click.echo(f"Error: Failed to start optimization - {e}", err=True)
                sys.exit(1)
    
    asyncio.run(make_request())


@optimize.command()
@click.argument('optimization_id')
@click.pass_context
def status(ctx, optimization_id):
    """Check optimization status."""
    
    api_url = ctx.obj['api_url']
    
    async def make_request():
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(f"{api_url}/api/v1/optimization/optimize/{optimization_id}")
                response.raise_for_status()
                result = response.json()
                
                click.echo(f"Optimization ID: {result['optimization_id']}")
                click.echo(f"Status: {result['status']}")
                
                if result['status'] == 'completed':
                    click.echo(f"Execution time: {result['execution_time']:.2f} seconds")
                    click.echo(f"Algorithm used: {result['algorithm_used']}")
                    click.echo(f"Mode used: {result['mode_used']}")
                    
                    if result['best_solution']:
                        click.echo("\nBest solution:")
                        click.echo(f"  ID: {result['best_solution']['id']}")
                        click.echo("  Objectives:")
                        for obj, value in result['best_solution']['objectives'].items():
                            click.echo(f"    {obj}: {value:.4f}")
                
                elif result['status'] == 'failed':
                    click.echo(f"Error: {result.get('error', 'Unknown error')}")
                
            except httpx.HTTPError as e:
                click.echo(f"Error: Failed to get optimization status - {e}", err=True)
                sys.exit(1)
    
    asyncio.run(make_request())


@optimize.command()
@click.argument('optimization_id')
@click.pass_context
def cancel(ctx, optimization_id):
    """Cancel a running optimization."""
    
    api_url = ctx.obj['api_url']
    
    async def make_request():
        async with httpx.AsyncClient() as client:
            try:
                response = await client.delete(f"{api_url}/api/v1/optimization/optimize/{optimization_id}")
                response.raise_for_status()
                result = response.json()
                
                click.echo(result['message'])
                
            except httpx.HTTPError as e:
                click.echo(f"Error: Failed to cancel optimization - {e}", err=True)
                sys.exit(1)
    
    asyncio.run(make_request())


@optimize.command()
@click.pass_context
def templates(ctx):
    """List available optimization templates."""
    
    api_url = ctx.obj['api_url']
    
    async def make_request():
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(f"{api_url}/api/v1/optimization/templates")
                response.raise_for_status()
                templates = response.json()
                
                click.echo("Available optimization templates:")
                click.echo()
                
                for template in templates:
                    click.echo(f"Name: {template['name']}")
                    click.echo(f"Description: {template['description']}")
                    click.echo(f"Objectives: {', '.join(template['objectives'])}")
                    click.echo(f"Variables: {', '.join(template['variables'])}")
                    click.echo(f"Use cases: {', '.join(template['use_cases'])}")
                    click.echo()
                
            except httpx.HTTPError as e:
                click.echo(f"Error: Failed to get templates - {e}", err=True)
                sys.exit(1)
    
    asyncio.run(make_request())


@cli.group()
def models():
    """Model management commands."""
    pass


@models.command()
@click.pass_context
def list(ctx):
    """List deployed models."""
    
    api_url = ctx.obj['api_url']
    
    async def make_request():
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(f"{api_url}/api/v1/inference/models")
                response.raise_for_status()
                models = response.json()
                
                click.echo("Deployed models:")
                click.echo()
                
                for model in models:
                    click.echo(f"ID: {model['model_id']}")
                    click.echo(f"Name: {model['name']}")
                    click.echo(f"Version: {model['version']}")
                    click.echo(f"Status: {model['status']}")
                    click.echo(f"Endpoint: {model.get('endpoint', 'N/A')}")
                    
                    if model.get('metrics'):
                        click.echo("Metrics:")
                        for metric, value in model['metrics'].items():
                            click.echo(f"  {metric}: {value}")
                    
                    click.echo()
                
            except httpx.HTTPError as e:
                click.echo(f"Error: Failed to list models - {e}", err=True)
                sys.exit(1)
    
    asyncio.run(make_request())


@models.command()
@click.argument('model_id')
@click.argument('inputs', type=click.File('r'))
@click.pass_context
def predict(ctx, model_id, inputs):
    """Make predictions using a model."""
    
    api_url = ctx.obj['api_url']
    
    try:
        input_data = json.load(inputs)
    except json.JSONDecodeError:
        click.echo("Error: Invalid JSON format in inputs file", err=True)
        sys.exit(1)
    
    async def make_request():
        async with httpx.AsyncClient() as client:
            try:
                data = {
                    'model_id': model_id,
                    'inputs': input_data
                }
                
                response = await client.post(f"{api_url}/api/v1/inference/predict", json=data)
                response.raise_for_status()
                result = response.json()
                
                click.echo("Prediction result:")
                click.echo(json.dumps(result, indent=2))
                
            except httpx.HTTPError as e:
                click.echo(f"Error: Failed to make prediction - {e}", err=True)
                sys.exit(1)
    
    asyncio.run(make_request())


@cli.command()
@click.pass_context
def health(ctx):
    """Check platform health."""
    
    api_url = ctx.obj['api_url']
    
    async def make_request():
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(f"{api_url}/health")
                response.raise_for_status()
                result = response.json()
                
                click.echo(f"Platform status: {result['status']}")
                click.echo(f"Version: {result['version']}")
                click.echo(f"Environment: {result['environment']}")
                click.echo(f"Uptime: {result['uptime']:.2f} seconds")
                
                click.echo("\nComponent health:")
                for component, health in result['checks'].items():
                    status_color = 'green' if health['status'] == 'healthy' else 'red'
                    click.echo(f"  {component}: ", nl=False)
                    click.secho(health['status'], fg=status_color)
                
            except httpx.HTTPError as e:
                click.echo(f"Error: Failed to check health - {e}", err=True)
                sys.exit(1)
    
    asyncio.run(make_request())


def main():
    """Main CLI entry point."""
    cli()


if __name__ == '__main__':
    main()
