//
// Created by tomas on 12/22/19.
//

#ifndef MUZERO_MAIN_H
#define MUZERO_MAIN_H

#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <random>
#include <thread>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/random_generator.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/program_options.hpp>
#include <boost/interprocess/sync/file_lock.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include "random.h"

#ifdef DEBUG
#define DUMP_LOG(msg) {std::cout << __PRETTY_FUNCTION__ << " (" << __LINE__ << ") " << msg << std::endl; std::fflush(stdout);}
#else
#define DUMP_LOG(msg)
#endif

const float MAXIMUM_FLOAT_VALUE = std::numeric_limits<float>::max();
const int ACTIONS = 31;
const int HISTORY = 8;
const int HIDDEN = 64;
//const std::string PRINTABLE = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ \t\n\r\x0b\x0c";

const std::string PRINTABLE = "abcdefghijklmnopqrstuvwxyz!,;. ";

typedef std::vector<std::shared_ptr<struct Node>> ChildrenList_t;
typedef std::vector<float> HiddenState_t;
typedef std::vector<struct Action> ActionList_t;
typedef std::vector<long> Image_t;
typedef std::vector<float> Policy_t;
typedef std::multimap<std::time_t, boost::filesystem::path> result_set_t;

struct NetworkOutput {
    float value;
    float reward;
    Policy_t policy_logits;
    HiddenState_t hidden_state;

    torch::Tensor value_tensor;
    torch::Tensor reward_tensor;
    torch::Tensor policy_tensor;
    torch::Tensor hidden_tensor;

    NetworkOutput(float value, float reward, Policy_t policy_logits, HiddenState_t hidden_state,
                  torch::Tensor value_tensor, torch::Tensor reward_tensor, torch::Tensor policy_tensor,
                  torch::Tensor hidden_tensor) :
            value(value), reward(reward), policy_logits(policy_logits), hidden_state(hidden_state),
            value_tensor(value_tensor), reward_tensor(reward_tensor), policy_tensor(policy_tensor),
            hidden_tensor(hidden_tensor) {

    }
};

template <typename T> struct Queue_i {
    virtual void enqueue(T) = 0;
    virtual T dequeue() = 0;
};

template <typename T>
struct Buffer : Queue_i<T> {
    std::list<T> buffer;
    std::mutex mutex;
    std::condition_variable cond;

    int max_size;

    Buffer(int max_size=10) : max_size(max_size), buffer({}) {}

    bool is_full() {
        return buffer.size() >= max_size;
    }

    bool is_empty() {
        return buffer.empty();
    }

    void enqueue(T inference) {
        std::unique_lock<std::mutex> lock(mutex);
        cond.wait(lock,[this] () {return !is_full();});
        buffer.emplace_back(inference);
        lock.unlock();
        cond.notify_all();

    }

    T dequeue() {
        std::unique_lock<std::mutex> lock(mutex);
        cond.wait(lock, [this]() {return !is_empty();});
        T result = buffer.front();
        buffer.pop_front();
        lock.unlock();
        cond.notify_all();
        return result;
    }
};

struct Action {
    int index;

    Action(int index=0) : index(index) {}

    friend std::ofstream& write (std::ofstream &out, const Action& obj) {
        out.write(reinterpret_cast<const char *>(&obj.index), sizeof(obj.index));
        return out;
    }

    friend std::ifstream &read (std::ifstream &in, Action& obj) {
        in.read(reinterpret_cast<char *>(&obj.index), sizeof(obj.index));
        return in;
    }
};

typedef struct {
    std::vector<NetworkOutput> out;
    NetworkOutput network_output(int idx=0) {
        return out[idx];
    }
} batch_out_t;

typedef Buffer<batch_out_t> batch_queue_out_t;

typedef struct batch_in_s {
    torch::Tensor batch;
    std::vector<long> actions;
    std::vector<std::shared_ptr<batch_queue_out_t>> out;

    static batch_in_s make_batch(HiddenState_t hidden_state, Action action) {
        batch_in_s ret;
        ret.batch = torch::tensor(hidden_state);
        ret.actions = {};
        ret.actions.emplace_back(action.index);
        return ret;
    }

    static batch_in_s make_batch(Image_t &t) {
        batch_in_s ret;
        ret.batch = torch::tensor(t).reshape({-1,static_cast<long>(t.size())});
        return ret;
    }
} batch_in_t;

typedef Buffer<batch_in_t> batch_queue_in_t;


typedef struct Inference_i {
    virtual batch_out_t initial_inference(batch_in_t batch) = 0;
    virtual batch_out_t recurrent_inference(batch_in_t batch) = 0;
} Inference_t;

typedef struct Network_i : Inference_i {
    virtual void save_network(int step, std::string path) = 0;
    virtual void load_network(std::string path) = 0;
    virtual int training_steps() = 0;
    virtual int inc_trainint_steps() = 0;
    virtual void _train(bool t=false) = 0;
    virtual std::vector<torch::Tensor> parameters() = 0;
    virtual void zero_grad() = 0;
} Network_t;

struct SharedStorage_i {
    virtual std::shared_ptr<Network_i> latest_network(torch::Device ctx) = 0;
    virtual void save_network(int step, std::shared_ptr<Network_i>& network) = 0;
};

#endif //MUZERO_MAIN_H
