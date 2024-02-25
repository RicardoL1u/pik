

from argparse import Namespace


def check_special_cases(args:Namespace):
    # for commonsenseqa, shot is 0, template must be mcq_cmqa
    if args.dataset == 'commonsense_qa':
        assert args.shot == 0, 'For commonsenseqa, shot must be 0'
        assert args.template == 'mcq_cmqa', 'For commonsenseqa, template must be mcq_cmqa'