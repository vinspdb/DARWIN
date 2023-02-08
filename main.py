from DARWIN import Darwin, DriftDiscoverMode
import argparse
if __name__ == '__main__':
    s = [DriftDiscoverMode.ADWIN, DriftDiscoverMode.STATIC]


    parser = argparse.ArgumentParser(description='DARWIN: An Online Deep Learning Approach to handle Concept Drifts in Predictive Process Monitoring')
    parser.add_argument('-event_log', type=str, help="Event log name")
    parser.add_argument('-model_update', type=int, help="0: Yes, 1:no")
    parser.add_argument('-strategy', type=str, help="FT - RT")

    args = parser.parse_args()

    Darwin.generate_csv(args.event_log)
    darwin = Darwin(
                    log_name=args.event_log,
                    drift_discover_algorithm=s[args.model_update],
                    model_update=str(s[args.model_update]),
                    train_strategy=args.strategy
                 )
    darwin.process_stream()

#python main.py -event_log bpi13closed -model_update 0 -strategy FT
