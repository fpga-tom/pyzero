/*
 *                     GNU GENERAL PUBLIC LICENSE
                       Version 3, 29 June 2007

 Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.

                            Preamble

  The GNU General Public License is a free, copyleft license for
software and other kinds of works.

  The licenses for most software and other practical works are designed
to take away your freedom to share and change the works.  By contrast,
the GNU General Public License is intended to guarantee your freedom to
share and change all versions of a program--to make sure it remains free
software for all its users.  We, the Free Software Foundation, use the
GNU General Public License for most of our software; it applies also to
any other work released this way by its authors.  You can apply it to
your programs, too.

  When we speak of free software, we are referring to freedom, not
price.  Our General Public Licenses are designed to make sure that you
have the freedom to distribute copies of free software (and charge for
them if you wish), that you receive source code or can get it if you
want it, that you can change the software or use pieces of it in new
free programs, and that you know you can do these things.

  To protect your rights, we need to prevent others from denying you
these rights or asking you to surrender the rights.  Therefore, you have
certain responsibilities if you distribute copies of the software, or if
you modify it: responsibilities to respect the freedom of others.

  For example, if you distribute copies of such a program, whether
gratis or for a fee, you must pass on to the recipients the same
freedoms that you received.  You must make sure that they, too, receive
or can get the source code.  And you must show them these terms so they
know their rights.

  Developers that use the GNU GPL protect your rights with two steps:
(1) assert copyright on the software, and (2) offer you this License
giving you legal permission to copy, distribute and/or modify it.

  For the developers' and authors' protection, the GPL clearly explains
that there is no warranty for this free software.  For both users' and
authors' sake, the GPL requires that modified versions be marked as
changed, so that their problems will not be attributed erroneously to
authors of previous versions.

  Some devices are designed to deny users access to install or run
modified versions of the software inside them, although the manufacturer
can do so.  This is fundamentally incompatible with the aim of
protecting users' freedom to change the software.  The systematic
pattern of such abuse occurs in the area of products for individuals to
use, which is precisely where it is most unacceptable.  Therefore, we
have designed this version of the GPL to prohibit the practice for those
products.  If such problems arise substantially in other domains, we
stand ready to extend this provision to those domains in future versions
of the GPL, as needed to protect the freedom of users.

  Finally, every program is threatened constantly by software patents.
States should not allow patents to restrict development and use of
software on general-purpose computers, but in those that do, we wish to
avoid the special danger that patents applied to a free program could
make it effectively proprietary.  To prevent this, the GPL assures that
patents cannot be used to render the program non-free.

  The precise terms and conditions for copying, distribution and
modification follow.
 */

#ifndef MUZERO_MAIN_H
#define MUZERO_MAIN_H

#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
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

struct NetworkOutputTensor {
    torch::Tensor value_tensor;
    torch::Tensor reward_tensor;
    torch::Tensor policy_tensor;
    torch::Tensor hidden_tensor;

    NetworkOutputTensor(torch::Tensor value_tensor, torch::Tensor reward_tensor, torch::Tensor policy_tensor,
                        torch::Tensor hidden_tensor) : value_tensor(value_tensor), reward_tensor(reward_tensor), policy_tensor(policy_tensor),
                                                       hidden_tensor(hidden_tensor) {

    }

    static std::shared_ptr<NetworkOutputTensor> make_tensor(torch::Tensor value_tensor, torch::Tensor reward_tensor, torch::Tensor policy_tensor,
                                                            torch::Tensor hidden_tensor) {
        return std::make_shared<NetworkOutputTensor>(value_tensor,  reward_tensor, policy_tensor,
                hidden_tensor);

    }
};

struct NetworkOutput {
    float value;
    float reward;
    Policy_t policy_logits;
    HiddenState_t hidden_state;

    std::shared_ptr<NetworkOutputTensor> tensor;


    NetworkOutput() {}

    NetworkOutput(float value, float reward, Policy_t policy_logits, HiddenState_t hidden_state,
                  std::shared_ptr<NetworkOutputTensor> network_output_tensor) :
            value(value), reward(reward), policy_logits(policy_logits), hidden_state(hidden_state),
            tensor(network_output_tensor) {

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

    std::vector<T> dequeue_all(std::chrono::milliseconds ms) {
        std::unique_lock<std::mutex> lock(mutex);
        if(cond.wait_for(lock, ms ,[this]() {return !is_empty();})) {
            std::vector<T> result = {};
            for (auto it = buffer.begin(); it != buffer.end(); ++it) {
                result.emplace_back(*it);
            }
            buffer.clear();
            lock.unlock();
            cond.notify_all();
            return result;
        }
        return {};
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

    static batch_in_s make_batch(HiddenState_t hidden_state, Action action, bool requires_grad=false) {
        batch_in_s ret;
        ret.batch = torch::tensor(hidden_state, torch::requires_grad(requires_grad));
        ret.actions = {};
        ret.actions.emplace_back(action.index);
        return ret;
    }

    static batch_in_s make_batch(torch::Tensor hidden_state, Action action, bool requires_grad=false) {
        batch_in_s ret;
        ret.batch = hidden_state.set_requires_grad(requires_grad).reshape({-1, HIDDEN});
        ret.actions = {};
        ret.actions.emplace_back(action.index);
        return ret;
    }

    static batch_in_s make_batch(torch::Tensor hidden_state, std::vector<Action> action, bool requires_grad=false) {
        batch_in_s ret;
        ret.batch = hidden_state.set_requires_grad(requires_grad).reshape({-1, HIDDEN});
        ret.actions = {};
        for(int i = 0;i < action.size(); i++) {
            ret.actions.emplace_back(action[i].index);
        }
        return ret;
    }

    static batch_in_s make_batch(Image_t &t) {
        batch_in_s ret;
        ret.batch = torch::tensor(t).reshape({-1,static_cast<long>(t.size())});
        return ret;
    }

    static batch_in_s make_batch(std::vector<Image_t> &t) {
        batch_in_s ret;
        ret.batch = torch::tensor(t[0]).reshape({-1,static_cast<long>(t[0].size())});
        for(int i = 1 ; i < t.size(); i++) {
            torch::Tensor tensor = torch::tensor(t[i]).reshape({-1,static_cast<long>(t[i].size())});
            ret.batch = torch::cat({ret.batch, tensor});
        }
        return ret;
    }
} batch_in_t;

typedef Buffer<batch_in_t> batch_queue_in_t;


typedef struct Inference_i {
    virtual batch_out_t initial_inference(batch_in_t& batch) = 0;
    virtual batch_out_t recurrent_inference(batch_in_t& batch) = 0;
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
