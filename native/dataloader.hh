#pragma once

#include <boost/callable_traits/args.hpp>
#include <memory>
#include <thread>
#include <type_traits>
#include <vector>

#include "blockingconcurrentqueue.h"

const int queue_size = 100;

template <typename F, typename G>
class Dataloader {
    using ResultType = std::remove_reference_t<
        std::tuple_element_t<1, boost::callable_traits::args_t<F>>>;
    using BatchType = std::result_of_t<G()>;
    using ResultPtr = std::unique_ptr<ResultType>;

   public:
    Dataloader(G _next_batch, F func, size_t num_threads)
        : next_batch(std::move(_next_batch)) {
        for (size_t i = 0; i < num_threads; i++) {
            threads.emplace_back([this, func]() {
                std::pair<BatchType, ResultPtr> next;

                while (true) {
                    batch_queue.wait_dequeue(next);

                    if (!next.first) {
                        // Done, shutting down
                        result_queue.enqueue(nullptr);
                        break;
                    } else {
                        func(*next.first, *next.second);
                        result_queue.enqueue(std::move(next.second));
                    }
                }
            });
        }
        remaining_threads = num_threads;

        for (size_t i = 0; i < num_threads * queue_size; i++) {
            recycle(std::make_unique<ResultType>());
        }
    }

    ~Dataloader() {
        BatchType b;
        for (size_t i = 0; i < threads.size(); i++) {
            batch_queue.enqueue(std::make_pair(b, nullptr));
        }

        for (auto& thread : threads) {
            thread.join();
        }
    }

    ResultPtr get_next() {
        ResultPtr next;
        while (remaining_threads > 0) {
            result_queue.wait_dequeue(next);
            if (next) {
                break;
            } else {
                remaining_threads--;
            }
        }
        return next;
    }

    void recycle(ResultPtr to_recycle) {
        batch_queue.enqueue(
            std::make_pair(next_batch(), std::move(to_recycle)));
    }

   private:
    G next_batch;
    int remaining_threads;
    std::vector<std::thread> threads;
    moodycamel::BlockingConcurrentQueue<std::pair<BatchType, ResultPtr>>
        batch_queue;
    moodycamel::BlockingConcurrentQueue<ResultPtr> result_queue;
};
