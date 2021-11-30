
import os
import time
import argparse
from my_strategies import ALL_STRATEGIES
from embeddings import ALL_EMBEDDINGS, ENCODER_NAME_TO_CLASS
from data_loader import load_train_test
from active_learning import ActiveLearner
from config import *
from utils import get_all_models


class Parser:
    @staticmethod
    def parse_args():
        parser = argparse.ArgumentParser(description='Hi')
        parser.add_argument('--name', type=str, required=False, default=DATASETS, choices=DATASETS,
                            help='dataset name. if not provided all datasets are used', nargs='+')
        parser.add_argument('--embedding', type=str, required=False, choices=ALL_EMBEDDINGS, default=ALL_EMBEDDINGS,
                            help='embedding name. if not provided all embeddings are used', nargs='+')
        parser.add_argument('--strategy', type=str, required=False, choices=ALL_STRATEGIES, default='unc',
                            help='strategy name. if not provided all enc are used', nargs='+')
        parser.add_argument('--model', type=str, required=False, choices=get_all_models(), default=get_all_models(), nargs='+')
        #parser.add_argument('--regs', type=str, required=False, choices=[str(x) for x in [0.5, 0.7, 0.9, 1.1, 1.3, 1.5]], default=[], nargs='+')
        parser.add_argument('--batch-size', type=float, required=False, default=0.5, help='batch size in % or samples')
        parser.add_argument('--skip', required=False, default=False, action='store_true', help='skip cfg if result exist')
        parser.add_argument('--downsample', required=False, default=-1)
        parser.add_argument('--round', type=int, required=False, default=0, help='start round')
        parser.add_argument('--normalization', required=False, default='l2', choices=['l2', 'l1'])
        parser.add_argument('--override', required=False, default=False, action='store_true')
        args = parser.parse_args()
        names = args.name
        encoders = args.embedding
        strategy = args.strategy
        models = args.model
        if type(names) != list:
            names = [names]
        if type(encoders) != list:
            encoders = [encoders]
        if type(strategy) != list:
            strategy = [strategy]
        if type(models) != list:
            models = [models]
        #if len(models) == 1 and len(args.regs):
        #    models = [models[0] + str(round(1/float(x), 2)).replace('.', ',') for x in args.regs]
        return names, encoders, strategy, args.round, models, float(args.downsample), args.normalization, args.batch_size, args.skip, args.override


def main():
    names, encoders, strategies, start_round, models, downsample, norm, batch_size, skip_existing, override = Parser.parse_args()
    a = time.time()
    for encoder_name in encoders:
        for dataset_name in names:
            encoder = ENCODER_NAME_TO_CLASS[encoder_name](norm=norm)
            dataset_name, X_train, X_test, y_train, y_test = load_train_test(dataset_name, encoder, downsample,override=override)

            learner = ActiveLearner(X_train, y_train, X_test, y_test, dataset_name, str(encoder), batch_size, skip_existing)
            for model_name in models:
                for strategy_name in strategies:
                    os.makedirs(os.path.join(CODE_DIR, 'results', str(encoder), dataset_name), exist_ok=True)
                    learner.run(strategy_name, model_name, start_round=start_round)
                    pass

    print('done in ', time.time()-a)


if __name__ == '__main__':
    #runtime_check()
    main()
