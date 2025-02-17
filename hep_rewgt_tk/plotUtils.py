import matplotlib.pyplot as plt

class PlotUtils:
    @staticmethod
    def plot_cv_results(cv_results, best_round):
        plt.figure(figsize=(10, 6))
        plt.plot(cv_results.index, cv_results['test-auc-mean'], label='Test AUC')
        plt.plot(cv_results.index, cv_results['train-auc-mean'], label='Train AUC')

        plt.fill_between(cv_results.index,
                        cv_results['test-auc-mean'] - cv_results['test-auc-std'],
                        cv_results['test-auc-mean'] + cv_results['test-auc-std'],
                        alpha=0.1)
        plt.fill_between(cv_results.index,
                        cv_results['train-auc-mean'] - cv_results['train-auc-std'],
                        cv_results['train-auc-mean'] + cv_results['train-auc-std'],
                        alpha=0.1)

        plt.axvline(x=best_round, color='r', linestyle='--', label=f'Best round ({best_round})')
        plt.xlabel('Number of Boosting Rounds')
        plt.ylabel('AUC Score')
        plt.title('AUC Score vs Number of Boosting Rounds')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        plt.show()
