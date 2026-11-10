"""Task-based dispatcher. Run from any directory using this script's path."""
import argparse
import runpy
import sys


COMMANDS = {
    'geometry': {'train': 'geometry.train', 'refine-surface': 'geometry.refine_surface',
                 'refine-scales': 'geometry.refine_scales', 'export-base': 'geometry.export_base',
                 'prepare-colmap': 'geometry.prepare_colmap'},
    'appearance': {'train': 'appearance.train'},
    'evaluation': {'render': 'evaluation.render', 'metrics': 'evaluation.metrics',
                   'verify-layers': 'evaluation.verify_layers'},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        epilog='Commands: ' + '; '.join(f'{task}: {", ".join(actions)}' for task, actions in COMMANDS.items()))
    parser.add_argument('task', choices=COMMANDS)
    parser.add_argument('action')
    parser.add_argument('arguments', nargs=argparse.REMAINDER, help='Arguments forwarded to the selected module')
    args = parser.parse_args()
    module = COMMANDS[args.task].get(args.action)
    if module is None:
        parser.error(f'Unknown {args.task} action {args.action!r}; choose from {list(COMMANDS[args.task])}')
    sys.argv = [module] + args.arguments
    runpy.run_module(module, run_name='__main__')


if __name__ == '__main__':
    main()
