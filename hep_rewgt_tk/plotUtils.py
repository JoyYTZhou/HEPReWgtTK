import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

class PlotUtils:
    @staticmethod
    def plot_cv_results(cv_results, best_round):
        fig, axes = plt.subplots(2, 1, figsize=(10, 12), sharex=True)  # Create 2 vertically stacked subplots
        
        # Plot AUC Scores
        axes[0].plot(cv_results.index, cv_results['test-auc-mean'], label='Test AUC')
        axes[0].plot(cv_results.index, cv_results['train-auc-mean'], label='Train AUC')

        axes[0].fill_between(cv_results.index,
                            cv_results['test-auc-mean'] - cv_results['test-auc-std'],
                            cv_results['test-auc-mean'] + cv_results['test-auc-std'],
                            alpha=0.1)
        axes[0].fill_between(cv_results.index,
                            cv_results['train-auc-mean'] - cv_results['train-auc-std'],
                            cv_results['train-auc-mean'] + cv_results['train-auc-std'],
                            alpha=0.1)

        axes[0].axvline(x=best_round, color='r', linestyle='--', label=f'Best round ({best_round})')
        axes[0].set_ylabel('AUC Score')
        axes[0].set_title('AUC Score vs Number of Boosting Rounds')
        axes[0].legend()
        axes[0].grid(True)

        # Plot Loss
        axes[1].plot(cv_results.index, cv_results['test-logloss-mean'], label='Test Log Loss')
        axes[1].plot(cv_results.index, cv_results['train-logloss-mean'], label='Train Log Loss')

        axes[1].fill_between(cv_results.index,
                            cv_results['test-logloss-mean'] - cv_results['test-logloss-std'],
                            cv_results['test-logloss-mean'] + cv_results['test-logloss-std'],
                            alpha=0.1)
        axes[1].fill_between(cv_results.index,
                            cv_results['train-logloss-mean'] - cv_results['train-logloss-std'],
                            cv_results['train-logloss-mean'] + cv_results['train-logloss-std'],
                            alpha=0.1)

        axes[1].axvline(x=best_round, color='r', linestyle='--', label=f'Best round ({best_round})')
        axes[1].set_xlabel('Number of Boosting Rounds')
        axes[1].set_ylabel('Log Loss')
        axes[1].set_title('Log Loss vs Number of Boosting Rounds')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()  # Adjust spacing between subplots
        plt.show()

