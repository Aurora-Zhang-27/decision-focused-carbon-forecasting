import argparse, os, json, matplotlib.pyplot as plt, numpy as np

def main(results_dir):
    path = os.path.join(results_dir, 'last_pred.txt')
    if not os.path.exists(path):
        print('No last_pred.txt found; run LSTM baseline first.')
        return
    yhat = np.loadtxt(path)
    plt.figure()
    plt.plot(yhat, label='prediction')
    plt.title('Last sequence prediction (LSTM baseline)')
    plt.legend()
    plt.tight_layout()
    out = os.path.join(results_dir, 'last_pred.png')
    plt.savefig(out)
    print('Saved', out)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', required=True)
    args = parser.parse_args()
    main(args.results_dir)
