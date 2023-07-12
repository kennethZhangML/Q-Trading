#include <iostream>
#include <vector>
#include <string>
#include <gym/gym.h>
#include <omp.h>

int main() {
    std::vector<std::string> stock_tickers = {"AAPL", "GOOGL", "MSFT", "AMZN"};
    std::vector<double> portfolio_weights = {0.3, 0.2, 0.3, 0.2};
    std::string start_date, end_date;

    std::cout << "Enter the start date (YYYY-MM-DD): ";
    std::cin >> start_date;
    std::cout << "Enter the end date (YYYY-MM-DD): ";
    std::cin >> end_date;

    gym::TradingEnv env(stock_tickers, portfolio_weights, start_date, end_date);
    auto observation = env.reset();
    int num_threads = omp_get_max_threads();
    bool done = false;

    #pragma omp parallel num_threads(num_threads)
    {
        int thread_id = omp_get_thread_num();
        int step = 0;

        while (!done) {
            int action = env.action_space.sample();
            auto result = env.step(action);
            observation = result.observation;
            double reward = result.reward;
            done = result.done;

            #pragma omp critical
            {
                std::cout << "Thread ID: " << thread_id << ", Step: " << step << std::endl;
                std::cout << "Observation: ";
                for (const auto& value : observation)
                    std::cout << value << " ";
                std::cout << std::endl;
                std::cout << "Reward: " << reward << std::endl;
                std::cout << "Done: " << (done ? "true" : "false") << std::endl;
            }

            if (thread_id == 0) {
                env.render();
            }
        
            step++;
            #pragma omp barrier

            #pragma omp single
            {
                if (done) {
                    break;
                }
            }
        }
    }

    env.close();

    return 0;
}
