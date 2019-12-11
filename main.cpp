#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <random>
#include <thread>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/random_generator.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <boost/filesystem.hpp>
#include "random.h"

torch::Device get_ctx() {
    return torch::Device("cpu");
}

torch::Device get_train_ctx() {
    return torch::Device("cpu");
}

const float MAXIMUM_FLOAT_VALUE = std::numeric_limits<float>::max();
const int ACTIONS = 100;
const int HISTORY = 8;
const int HIDDEN = 128;
const std::string PRINTABLE = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ \t\n\r\x0b\x0c";

typedef std::vector<std::shared_ptr<struct Node>> ChildrenList_t;
typedef std::vector<float> HiddenState_t;
typedef std::vector<struct Action> ActionList_t;
typedef std::vector<long> Image_t;
typedef std::vector<float> Policy_t;

Random rnd(Random::kUniqueSeed, Random::kUniqueStream);

struct Game;

struct KnownBounds {
    KnownBounds(float min = -MAXIMUM_FLOAT_VALUE, float max = MAXIMUM_FLOAT_VALUE) : min(min), max(max) {

    }

    float min;
    float max;
};

class MinMaxStats {
    float maximum;
    float minimum;
public:
    MinMaxStats(const struct KnownBounds &knownBounds) : maximum(knownBounds.max), minimum(knownBounds.min) {

    }

    void update(float value) {
        maximum = std::max(maximum, value);
        minimum = std::min(minimum, value);
    }

    float normalize(float value) {
        if (maximum > minimum) {
            return (value - minimum) / (maximum - minimum);
        };
        return value;
    }
};

class VisitSoftmaxTemperatureFn {
public:
    virtual float operator()(int num_moves, int training_steps) = 0;
};

struct MuZeroConfig {
    MuZeroConfig(
            int action_space_size,
            int max_moves,
            float discount,
            float dirichlet_alpha,
            int num_simulations,
            int batch_size,
            int td_steps,
            int num_actors,
            float lr_init,
            float lr_decay_steps,
            VisitSoftmaxTemperatureFn *visit_softmax_temperature_fn,
            KnownBounds known_bounds
    ) :
            action_space_size(action_space_size),
            num_actors(num_actors),

            visit_softmax_temperature_fn(visit_softmax_temperature_fn),
            max_moves(max_moves),
            num_simulations(num_simulations),
            discount(discount),

            // Root prior exploration noise.
            root_dirichlet_alpha(dirichlet_alpha),
            root_exploration_fraction(0.25),

            // UCB formula
            pb_c_base(19652),
            pb_c_init(1.25),

            // If we already have some information about which values occur in the
            // environment, we can use them to initialize the rescaling.
            // This is not strictly necessary, but establishes identical behaviour to
            // AlphaZero in board games.
            known_bounds(known_bounds),

            // Training
            training_steps(int(1000e3)),
            checkpoint_interval(int(1)),
            window_size(int(1e6)),
            batch_size(batch_size),
            num_unroll_steps(5),
            td_steps(td_steps),

            weight_decay(1e-4),
            momentum(0.9),

            // Exponential learning rate schedule
            lr_init(lr_init),
            lr_decay_rate(0.1),
            lr_decay_steps(lr_decay_steps) {

    }

    // Self-Play
    int action_space_size;
    int num_actors;

    VisitSoftmaxTemperatureFn* visit_softmax_temperature_fn;
    int max_moves;
    int num_simulations;
    float discount;

    // Root prior exploration noise.
    float root_dirichlet_alpha;
    float root_exploration_fraction;

    // UCB formula
    float pb_c_base;
    float pb_c_init;

    // If we already have some information about which values occur in the
    // environment, we can use them to initialize the rescaling.
    // This is not strictly necessary, but establishes identical behaviour to
    // AlphaZero in board games.
    KnownBounds known_bounds;

    // Training
    int training_steps;
    int checkpoint_interval;
    int window_size;
    int batch_size;
    int num_unroll_steps;
    int td_steps;

    float weight_decay;
    float momentum;

    // Exponential learning rate schedule
    float lr_init;
    float lr_decay_rate;
    float lr_decay_steps;

    std::shared_ptr<Game> new_game() {
        return std::make_shared<Game>(action_space_size, discount);
    }
};

MuZeroConfig make_board_config(int action_space_size, int max_moves,
                               float dirichlet_alpha,
                               float lr_init) {

    class VisitSoftmaxTemperatureFn1 : public VisitSoftmaxTemperatureFn {
    public:
        float operator()(int num_moves, int training_steps) override {
            if (num_moves < 30)
                return 1.0;
            else
                return 0.0;
        }
    };

    return {
            action_space_size,
            max_moves, 1.0,
            dirichlet_alpha,
            150,
            128,
            max_moves,  //Always use Monte Carlo return.
            8,
            lr_init,
            400e3,
            new VisitSoftmaxTemperatureFn1(),
            KnownBounds(-1, 1)
    };
}

MuZeroConfig make_c_config() {
    return make_board_config(ACTIONS, 16, 0.03, 0.001);
}

struct Action {
    int index;

    Action(int index=0) : index(index) {}

    friend std::ofstream &operator <<(std::ofstream &out, const Action& obj) {
        out.write(reinterpret_cast<const char *>(&obj.index), sizeof(obj.index));
        return out;
    }

    friend std::ifstream &operator>>(std::ifstream &in, Action& obj) {
        in.read(reinterpret_cast<char *>(&obj.index), sizeof(obj.index));
        return in;
    }};

struct Player {
    int id = -1;

};


struct Node {
    int visit_count;
    int to_play;
    float prior;
    float value_sum;
    ChildrenList_t children;
    HiddenState_t hidden_state;
    float reward;

    explicit Node(float prior) :
            visit_count(0),
            to_play(-1),
            prior(prior),
            value_sum(0),
            children({}),
            hidden_state({}),
            reward(0) {

    }

    float value() {
        if(visit_count == 0) {
            return 0;
        } else {
            return value_sum / (float)visit_count;
        }
    }

    bool expanded() {
        return !children.empty();
    }

};

struct ActionHistory {
    ActionList_t history;
    int action_space_size;

    ActionHistory(ActionList_t &history, int action_space_size) :
            history(history), action_space_size(action_space_size) {

    }

    ActionHistory clone() {
        return {history, action_space_size};
    }

    void add_action(Action &action) {
        history.emplace_back(action);
    }

    Action last_action() {
        return history[history.size() - 1];
    }

    ActionList_t action_space() {
        ActionList_t r;
        for (int i = 0; i < action_space_size; i++) {
            r.emplace_back(Action(i));
        }
        return r;
    }

    static Player to_play() {
        return {};
    }
};

template <typename T>
std::ofstream &operator <<(std::ofstream &out, const T& obj) {
    out.write(reinterpret_cast<const char *>(&obj), sizeof(T));
    return out;
}

template <typename T>
std::ifstream &operator >>(std::ifstream &in, const T& obj) {
    in.read((std::ifstream::char_type *)(&obj), sizeof(T));
    return in;
}

template <typename T>
std::ofstream &operator <<(std::ofstream &out, const std::vector<T>& obj) {
    typename std::vector<T>::size_type count = obj.size();
    out.write(reinterpret_cast<const char *>(&count), sizeof(count));
    out.write(reinterpret_cast<const char *>(obj.data()), sizeof(T) * obj.size());
    return out;
}

template <typename T>
std::ofstream &operator <<(std::ofstream &out, const std::vector<std::vector<T>>& obj) {
    typename std::vector<T>::size_type count = obj.size();
    out.write(reinterpret_cast<const char *>(&count), sizeof(count));
    for(typename std::vector<T>::size_type i = 0; i < obj.size(); i++)
        out << obj[i];
    return out;
}

template <typename T>
std::ifstream &operator>>(std::ifstream &in, std::vector<T>& obj) {
    typename std::vector<T>::size_type count=0;
    in.read(reinterpret_cast<char *>(&count), sizeof(count));
    for(typename std::vector<T>::size_type i = 0; i < count; i++) {
        T a;
        in.read(reinterpret_cast<char *>(&a), sizeof(T));
        obj.emplace_back(a);
    }
//    std::copy_n(std::istream_iterator<T>(in), count, std::back_inserter(obj));
    return in;
}

std::ifstream &operator>>(std::ifstream &in, ActionList_t & obj) {
    ActionList_t::size_type count=0;
    in.read(reinterpret_cast<char *>(&count), sizeof(count));

    for(int i = 0; i < count; i++) {
        Action a;
        in >> a;
        obj.emplace_back(a);
    }
    return in;
}

std::ofstream &operator<<(std::ofstream &out, ActionList_t & obj) {
    ActionList_t::size_type count=obj.size();
    out.write(reinterpret_cast<char *>(&count), sizeof(count));

    for(ActionList_t::size_type i = 0; i < count; i++) {
        out << obj[i];
    }
    return out;
}

template <typename T>
std::ifstream &operator>>(std::ifstream &in, std::vector<std::vector<T>>& obj) {
    typename std::vector<T>::size_type count = 0;
    in.read(reinterpret_cast<char *>(&count), sizeof(count));
    for(typename std::vector<T>::size_type i = 0; i < count; i++) {
        std::vector<T> obj1;
        in >> obj1;
        obj.emplace_back(obj1);
    }
    return in;
}

struct Environment {

    std::ifstream f;
    std::vector<long> seq;

    Environment() : seq({}) {
        f.open("/home/tomas/CLionProjects/muzero/example.txt");
        if (!f) {
            std::cerr << "unable to open example.txt" << std::endl;
        }
    }

    int step(Action &action) {
        std::string line;
        seq.emplace_back(action.index);
        std::stringstream result;
        for(int i = 0; i < seq.size(); i++) {
            result << PRINTABLE[seq[i]];
        }
//        std::copy(seq.begin(), seq.end(), std::ostream_iterator<int>(result, ""));
        std::string str = result.str();
        std::cout << str << std::endl;
        f.seekg(0, std::ios::beg);
        while (f.good()) {
            getline(f, line);
            size_t pos = line.find(str);
            if (pos != std::string::npos) {
                return 1;
            }
        }
        return -1;
    }

    virtual ~Environment() {
        f.close();
    }

    static ActionList_t get_actions() {
        ActionList_t r;
        for (int i = 0; i < ACTIONS; i++) {
            r.emplace_back(Action(i));
        }
        return r;
    }

    friend std::ofstream &operator <<(std::ofstream &out, const Environment& obj) {
        out << obj.seq;
        return out;
    }

    friend std::ifstream &operator>>(std::ifstream &in, Environment& obj) {
        in >> obj.seq;
        return in;
    }

};

struct Target {
    float value;
    float reward;
    Policy_t policy;
    Target(float value, float reward, Policy_t policy) : value(value), reward(reward), policy(policy) {

    }
};

struct Game {

    Environment environment;  // Game specific environment.
    ActionList_t history;
    std::vector<float> rewards;
    std::vector<std::vector<float>> child_visits;
    std::vector<float> root_values;
    int action_space_size;
    float discount;

    Game(int action_space_size, float discount) :
            history({}),
            rewards({}),
            child_visits({}),
            root_values({}),
            action_space_size(action_space_size),
            discount(discount) {
    };

    bool terminal() {
        if(history.size() > 0) {
            return rewards[rewards.size()-1] == -1;
        }
        return false;
    }

    ActionList_t legal_actions() {
        return environment.get_actions();
    }

    void apply(Action& action) {
        float reward = environment.step(action);
        rewards.emplace_back(reward);
        history.emplace_back(action);
    }

    void store_search_statistics(std::shared_ptr<Node>& root) {
        int sum_visits = std::accumulate(root->children.begin(), root->children.end(), 0,
                [](int a,std::shared_ptr<struct Node>& b){ return a + b->visit_count;});
        std::vector<float> v;
//        std::cout << "\t\t\t\t\t sum_visits: " << sum_visits << std::endl;
        for(int i = 0; i < ACTIONS; i++) {
            // TODO: check if all children nodes are expanded already
            float cv = (float) root->children[i]->visit_count / (float) sum_visits;
//            std::cout << cv << std::endl;
            v.emplace_back(cv);
        }
        child_visits.emplace_back(v);
        root_values.emplace_back(root->value());
    }

    Image_t make_image(int state_index) {
        Image_t r;
        state_index = state_index < 0 ? environment.seq.size() + state_index : state_index;
        int low = std::max(state_index - HISTORY, 0);
        std::copy(environment.seq.begin() + low, environment.seq.begin() + state_index + 1, std::back_inserter(r));
        int pad = HISTORY - r.size();
        for(int i = 0;i < pad; i++) {
            r.emplace_back(0);
        }
        return r;
    }

    std::vector<Target> make_target(int state_index, int num_unroll_steps, int td_steps, Player to_play) {
        std::vector<Target> targets;
        for(int current_index = state_index; current_index < state_index + num_unroll_steps+1; current_index++) {
            int bootstrap_index = current_index + td_steps;
            float value = 0;
            if (bootstrap_index < root_values.size()) {
                value = root_values[bootstrap_index]*std::pow(discount,(float)td_steps);
            } else {
                value = 0;
            }

            assert(!isnan(value));

            for(int i = current_index; i < std::min(bootstrap_index, (int)rewards.size()); i++) {
                value += rewards[i] * std::pow(discount, (float)(i-current_index));
            }

            assert(!isnan(value));

            if(current_index < root_values.size()) {
                assert(child_visits[current_index].size() > 0);
                for(int i = 0; i < child_visits[current_index].size(); i++) {
                    assert(!isnan(child_visits[current_index][i]));
                }
                targets.emplace_back(Target(value, rewards[current_index], child_visits[current_index]));
            } else {
                targets.emplace_back(Target(0, 0, {}));
            }
        }
        return targets;
    }


    Player to_play() {
        return {};
    }

    friend std::ofstream &operator <<(std::ofstream &out, const Game& obj) {
        out << obj.environment
         << obj.history
         << obj.rewards
         << obj.child_visits
         << obj.root_values
         << (int)obj.action_space_size
         << (int)obj.discount;
        return out;
    }

    friend std::ifstream &operator>>(std::ifstream &in, Game& obj) {
        in >> obj.environment
        >> obj.history
        >> obj.rewards
        >> obj.child_visits
        >> obj.root_values
        >> (int)obj.action_space_size
        >> (int)obj.discount;
        return in;
    }

    void save(const std::string& filename) {
        std::string path = "/home/tomas/CLionProjects/muzero/replay_buffer/" + filename;
        assert(!boost::filesystem::is_regular_file(path));
        std::ofstream out(path, std::ios::out | std::ios::binary);
        assert(history.size() > 0);
        assert(child_visits.size() > 0);
        assert(child_visits[0].size() > 0);
        for(int i = 0; i < child_visits.size(); i++) {
            for(int j = 0; j < child_visits[i].size(); j++)
                assert(!isnan(child_visits[i][j]));
        }
        out << *this;
        out.close();
        assert(boost::filesystem::is_regular_file(path));
    }

    void load(const std::string& filename) {
//        std::cout << filename << std::endl;
        assert(boost::filesystem::is_regular_file(filename));
        std::ifstream in(filename, std::ios::in | std::ios::binary);
        in >> *this;
        in.close();
        assert(environment.seq.size() > 0);
        assert(history.size() > 0);
        assert(child_visits.size() > 0);
        assert(child_visits[0].size() > 0);

        for(int i = 0; i < child_visits.size(); i++) {
            for(int j = 0; j < child_visits[i].size(); j++)
                assert(!isnan(child_visits[i][j]));
        }
    }

    ActionHistory action_history() {
        return {history, action_space_size};
    }
};

struct Batch {
    Image_t image;
    ActionList_t action;
    std::vector<Target> target;
    Batch(Image_t image, ActionList_t action, std::vector<Target> target) :
        image(image), action(action), target(target) {

    }

};

struct ReplayBuffer {
    int window_size;
    int batch_size;
    std::vector<std::shared_ptr<Game>> buffer;
    ReplayBuffer(MuZeroConfig& config) : window_size(config.window_size), batch_size(config.batch_size),
        buffer({}) {
    }

    void save_game(std::shared_ptr<Game> game) {
        if(buffer.size() > window_size) {
            buffer.erase(buffer.begin());
        }
        buffer.emplace_back(game);
        boost::uuids::random_generator gen;
        boost::uuids::uuid u = gen();
        game->save(boost::uuids::to_string(u));
    }

    std::vector<Batch> sample_batch(int num_unroll_steps, int td_steps) {
        std::vector<Batch> result;
        std::vector<std::shared_ptr<Game>> games;
        for(int i = 0; i < batch_size; i++) {
            games.emplace_back(sample_game());
        }
        std::vector<int> game_pos;
        for(int i = 0; i < games.size(); i++) {
            game_pos.emplace_back(sample_position(games[i]));
        }
        for(int i = 0; i < game_pos.size(); i++) {
            auto g = games[i];
            Image_t image = g->make_image(game_pos[i]);
            ActionList_t history;
            std::copy(g->history.begin()+game_pos[i], g->history.begin()+std::min(game_pos[i]+num_unroll_steps, (int)g->history.size()), std::back_inserter(history));
            std::vector<Target> target = g->make_target(game_pos[i], num_unroll_steps, td_steps, g->to_play());
            result.emplace_back(Batch{image, history, target});
        }
        return result;
    }

    std::shared_ptr<Game> sample_game() {
        std::string path = "/home/tomas/CLionProjects/muzero/replay_buffer/";
        while(!boost::filesystem::is_directory(path)) {
            sleep(5);
        }


        while (boost::filesystem::is_empty(path)) {
            sleep(5);
        }

        int cnt = std::count_if(
        boost::filesystem::directory_iterator(path),
        boost::filesystem::directory_iterator(),
                static_cast<bool(*)(const boost::filesystem::path&)>(boost::filesystem::is_regular_file) );


        std::random_device seeder;
        std::mt19937 engine(seeder());
        std::uniform_int_distribution<int> dist(0, cnt-1);
        int guess = dist(engine);


        for(boost::filesystem::directory_iterator it(path); it != boost::filesystem::directory_iterator(); ++it) {
            if(guess == 0) {
                auto game = std::make_shared<Game>(0, 0);
                game->load(it->path().c_str());
                return game;
            } else {
                guess--;
            }
        }
        assert(false);
    }

    int sample_position(std::shared_ptr<Game>& game) {
        std::random_device seeder;
        std::mt19937 engine(seeder());
        std::uniform_int_distribution<int> dist(0, game->history.size()-1);
        int guess = dist(engine);
        return guess;
    }
};

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
torch::nn::Conv2dOptions conv_options(int64_t in_planes, int64_t out_planes, int64_t kerner_size,
                                      int64_t stride=1, int64_t padding=0, bool with_bias=false){
    torch::nn::Conv2dOptions conv_options = torch::nn::Conv2dOptions(in_planes, out_planes, kerner_size);
    conv_options.stride(stride);
    conv_options.padding(padding);
    conv_options.with_bias(with_bias);
    return conv_options;
}


struct BasicBlock : torch::nn::Module {

    static const int expansion;

    int64_t stride;
    torch::nn::Conv2d conv1;
    torch::nn::BatchNorm bn1;
    torch::nn::Conv2d conv2;
    torch::nn::BatchNorm bn2;
    torch::nn::Sequential downsample;

    BasicBlock(int64_t inplanes, int64_t planes, int64_t stride_=1,
               torch::nn::Sequential downsample_=torch::nn::Sequential())
            : conv1(conv_options(inplanes, planes, 3, stride_, 1)),
              bn1(planes),
              conv2(conv_options(planes, planes, 3, 1, 1)),
              bn2(planes),
              downsample(downsample_)
    {
        register_module("conv1", conv1);
        register_module("bn1", bn1);
        register_module("conv2", conv2);
        register_module("bn2", bn2);
        stride = stride_;
        if (!downsample->is_empty()){
            register_module("downsample", downsample);
        }
    }

    torch::Tensor forward(torch::Tensor x) {
        at::Tensor residual(x.clone());

        x = conv1->forward(x);
        x = bn1->forward(x);
        x = torch::relu(x);

        x = conv2->forward(x);
        x = bn2->forward(x);

        if (!downsample->is_empty()){
            residual = downsample->forward(residual);
        }

        x += residual;
        x = torch::relu(x);

        return x;
    }
};

const int BasicBlock::expansion = 1;


struct BottleNeck : torch::nn::Module {

    static const int expansion;

    int64_t stride;
    torch::nn::Conv2d conv1;
    torch::nn::BatchNorm bn1;
    torch::nn::Conv2d conv2;
    torch::nn::BatchNorm bn2;
    torch::nn::Conv2d conv3;
    torch::nn::BatchNorm bn3;
    torch::nn::Sequential downsample;

    BottleNeck(int64_t inplanes, int64_t planes, int64_t stride_=1,
               torch::nn::Sequential downsample_=torch::nn::Sequential())
            : conv1(conv_options(inplanes, planes, 1)),
              bn1(planes),
              conv2(conv_options(planes, planes, 3, stride_, 1)),
              bn2(planes),
              conv3(conv_options(planes, planes * expansion , 1)),
              bn3(planes * expansion),
              downsample(downsample_)
    {
        register_module("conv1", conv1);
        register_module("bn1", bn1);
        register_module("conv2", conv2);
        register_module("bn2", bn2);
        register_module("conv3", conv3);
        register_module("bn3", bn3);
        stride = stride_;
        if (!downsample->is_empty()){
            register_module("downsample", downsample);
        }
    }

    torch::Tensor forward(torch::Tensor x) {
        at::Tensor residual(x.clone());

        x = conv1->forward(x);
        x = bn1->forward(x);
        x = torch::relu(x);

        x = conv2->forward(x);
        x = bn2->forward(x);
        x = torch::relu(x);

        x = conv3->forward(x);
        x = bn3->forward(x);

        if (!downsample->is_empty()){
            residual = downsample->forward(residual);
        }

        x += residual;
        x = torch::relu(x);

        return x;
    }
};

const int BottleNeck::expansion = 4;


template <class Block> struct ResNet_representation : torch::nn::Module {

    int64_t inplanes = 64;
    torch::nn::Conv2d conv1;
    torch::nn::BatchNorm bn1;
    torch::nn::Sequential layer1;
    torch::nn::Sequential layer2;
    torch::nn::Sequential layer3;
    torch::nn::Sequential layer4;
    torch::nn::Linear fc;

    ResNet_representation(torch::IntList layers, int64_t num_classes=1000)
            : conv1(conv_options(8, 64, 3, 1, 1)),
              bn1(64),
              layer1(_make_layer(64, layers[0])),
              layer2(_make_layer(128, layers[1], 2)),
              layer3(_make_layer(256, layers[2], 2)),
              layer4(_make_layer(512, layers[3], 2)),
              fc(512 * Block::expansion, num_classes)
    {
        register_module("conv1", conv1);
        register_module("bn1", bn1);
        register_module("layer1", layer1);
        register_module("layer2", layer2);
        register_module("layer3", layer3);
        register_module("layer4", layer4);
        register_module("fc", fc);

        // Initializing weights
//        for(auto m: this->modules()){
//            if (m->name() == "torch::nn::Conv2dImpl"){
//                for (auto p: m->parameters()){
//                    torch::nn::init::xavier_normal_(p.values());
//                }
//            }
//            else if (m->name() == "torch::nn::BatchNormImpl"){
//                for (auto p: m->parameters()){
//                    if (p. == "weight"){
//                        torch::nn::init::constant_(p.value, 1);
//                    }
//                    else if (p.key == "bias"){
//                        torch::nn::init::constant_(p.value, 0);
//                    }
//                }
//            }
//        }
    }

    torch::Tensor forward(torch::Tensor x){

        x = conv1->forward(x);
        x = bn1->forward(x);
        x = torch::relu(x);
        x = torch::max_pool2d(x, 3, 1, 1);

        x = layer1->forward(x);
        x = layer2->forward(x);
        x = layer3->forward(x);
        x = layer4->forward(x);

        x = torch::avg_pool2d(x, 3, 3, 1);
        x = x.view({x.sizes()[0], -1});
        x = fc->forward(x);

        return x;
    }


private:
    torch::nn::Sequential _make_layer(int64_t planes, int64_t blocks, int64_t stride=1){
        torch::nn::Sequential downsample;
        if (stride != 1 or inplanes != planes * Block::expansion){
            downsample = torch::nn::Sequential(
                    torch::nn::Conv2d(conv_options(inplanes, planes * Block::expansion, 1, stride)),
                    torch::nn::BatchNorm(planes * Block::expansion)
            );
        }
        torch::nn::Sequential layers;
        layers->push_back(Block(inplanes, planes, stride, downsample));
        inplanes = planes * Block::expansion;
        for (int64_t i = 0; i < blocks; i++){
            layers->push_back(Block(inplanes, planes));
        }

        return layers;
    }
};

template <class Block> struct ResNet_dynamics : torch::nn::Module {

    int64_t inplanes = 64;
    torch::nn::Conv2d conv1;
    torch::nn::BatchNorm bn1;
    torch::nn::Sequential layer1;
    torch::nn::Sequential layer2;
    torch::nn::Sequential layer3;
    torch::nn::Sequential layer4;
    torch::nn::Linear fc;
    torch::nn::Linear fc1;

    ResNet_dynamics(torch::IntList layers, int64_t num_classes=1000)
            : conv1(conv_options(32, 64, 3, 1, 1)),
              bn1(64),
              layer1(_make_layer(64, layers[0])),
              layer2(_make_layer(128, layers[1], 1)),
              layer3(_make_layer(256, layers[2], 1)),
              layer4(_make_layer(512, layers[3], 2)),
              fc(512 * Block::expansion, num_classes),
              fc1(512 * Block::expansion, 1)
    {
        register_module("conv1", conv1);
        register_module("bn1", bn1);
        register_module("layer1", layer1);
        register_module("layer2", layer2);
        register_module("layer3", layer3);
        register_module("layer4", layer4);
        register_module("fc", fc);
        register_module("fc1", fc1);

        // Initializing weights
//        for(auto m: this->modules()){
//            if (m->name() == "torch::nn::Conv2dImpl"){
//                for (auto p: m->parameters()){
//                    torch::nn::init::xavier_normal_(p.values());
//                }
//            }
//            else if (m->name() == "torch::nn::BatchNormImpl"){
//                for (auto p: m->parameters()){
//                    if (p. == "weight"){
//                        torch::nn::init::constant_(p.value, 1);
//                    }
//                    else if (p.key == "bias"){
//                        torch::nn::init::constant_(p.value, 0);
//                    }
//                }
//            }
//        }
    }

    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x){

        x = conv1->forward(x.reshape({-1,32,HIDDEN/32,1}));
        x = bn1->forward(x);
        x = torch::relu(x);
        x = torch::max_pool2d(x, 3, 1, 1);

        x = layer1->forward(x);
        x = layer2->forward(x);
        x = layer3->forward(x);
        x = layer4->forward(x);

        x = torch::avg_pool2d(x, 3, 3, 1);
        torch::Tensor y = x.view({x.sizes()[0], -1});
        x = fc->forward(y);
        y = torch::tanh(fc1->forward(y));

        return {x,y};
    }


private:
    torch::nn::Sequential _make_layer(int64_t planes, int64_t blocks, int64_t stride=1){
        torch::nn::Sequential downsample;
        if (stride != 1 or inplanes != planes * Block::expansion){
            downsample = torch::nn::Sequential(
                    torch::nn::Conv2d(conv_options(inplanes, planes * Block::expansion, 1, stride)),
                    torch::nn::BatchNorm(planes * Block::expansion)
            );
        }
        torch::nn::Sequential layers;
        layers->push_back(Block(inplanes, planes, stride, downsample));
        inplanes = planes * Block::expansion;
        for (int64_t i = 0; i < blocks; i++){
            layers->push_back(Block(inplanes, planes));
        }

        return layers;
    }
};

template <class Block> struct ResNet_prediction : torch::nn::Module {

    int64_t inplanes = 64;
    torch::nn::Conv2d conv1;
    torch::nn::BatchNorm bn1;
    torch::nn::Sequential layer1;
    torch::nn::Sequential layer2;
    torch::nn::Sequential layer3;
    torch::nn::Sequential layer4;
    torch::nn::Linear fc;
    torch::nn::Linear fc1;

    ResNet_prediction(torch::IntList layers, int64_t num_classes=1000)
            : conv1(conv_options(32, 64, 3, 1, 1)),
              bn1(64),
              layer1(_make_layer(64, layers[0])),
              layer2(_make_layer(128, layers[1], 1)),
              layer3(_make_layer(256, layers[2], 1)),
              layer4(_make_layer(512, layers[3], 2)),
              fc(512 * Block::expansion, num_classes),
              fc1(512 * Block::expansion, 1)
    {
        register_module("conv1", conv1);
        register_module("bn1", bn1);
        register_module("layer1", layer1);
        register_module("layer2", layer2);
        register_module("layer3", layer3);
        register_module("layer4", layer4);
        register_module("fc", fc);
        register_module("fc1", fc1);

        // Initializing weights
//        for(auto m: this->modules()){
//            if (m->name() == "torch::nn::Conv2dImpl"){
//                for (auto p: m->parameters()){
//                    torch::nn::init::xavier_normal_(p.values());
//                }
//            }
//            else if (m->name() == "torch::nn::BatchNormImpl"){
//                for (auto p: m->parameters()){
//                    if (p. == "weight"){
//                        torch::nn::init::constant_(p.value, 1);
//                    }
//                    else if (p.key == "bias"){
//                        torch::nn::init::constant_(p.value, 0);
//                    }
//                }
//            }
//        }
    }

    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x){

        x = conv1->forward(x.reshape({-1,32,HIDDEN/32,1}));
        x = bn1->forward(x);
        x = torch::relu(x);
        x = torch::max_pool2d(x, 3, 1, 1);

        x = layer1->forward(x);
        x = layer2->forward(x);
        x = layer3->forward(x);
        x = layer4->forward(x);

        x = torch::avg_pool2d(x, 3, 3, 1);
        torch::Tensor y = x.view({x.sizes()[0], -1});
        x = fc->forward(y);
        y = torch::tanh(fc1->forward(y));

        return {x, y};
    }


private:
    torch::nn::Sequential _make_layer(int64_t planes, int64_t blocks, int64_t stride=1){
        torch::nn::Sequential downsample;
        if (stride != 1 or inplanes != planes * Block::expansion){
            downsample = torch::nn::Sequential(
                    torch::nn::Conv2d(conv_options(inplanes, planes * Block::expansion, 1, stride)),
                    torch::nn::BatchNorm(planes * Block::expansion)
            );
        }
        torch::nn::Sequential layers;
        layers->push_back(Block(inplanes, planes, stride, downsample));
        inplanes = planes * Block::expansion;
        for (int64_t i = 0; i < blocks; i++){
            layers->push_back(Block(inplanes, planes));
        }

        return layers;
    }
};


std::shared_ptr<ResNet_representation<BasicBlock>> resnet_representation(int num_classes){
    return std::make_shared<ResNet_representation<BasicBlock>>(torch::IntList ({2, 2, 2, 2}), num_classes);
}

std::shared_ptr<ResNet_dynamics<BasicBlock>> resnet_dynamics(int num_classes){
    return std::make_shared<ResNet_dynamics<BasicBlock>>(torch::IntList ({2, 2, 2, 2}), num_classes);
}

std::shared_ptr<ResNet_prediction<BasicBlock>> resnet_prediction(int num_classes){
    return std::make_shared<ResNet_prediction<BasicBlock>>(torch::IntList ({2, 2, 2, 2}), num_classes);
}


struct Representation : torch::nn::Module {
    torch::Device ctx;
    std::shared_ptr<ResNet_representation<BasicBlock>> network;
    torch::nn::Embedding embedding;
    Representation(torch::Device& ctx) : ctx(ctx), network(resnet_representation(HIDDEN)),
        embedding(torch::nn::Embedding(ACTIONS, 20)) {
        register_module("embedding", embedding);
        register_module("representation", network);
        network->to(ctx);
    }

    torch::Tensor forward(torch::Tensor x) {
        auto emb = embedding(x);
        return network->forward(emb.reshape({-1,HISTORY, 1, 20}));
    }
};

struct Prediction : torch::nn::Module {
    torch::Device ctx;
    std::shared_ptr<ResNet_prediction<BasicBlock>> network;
    Prediction(torch::Device& ctx) : ctx(ctx), network(resnet_prediction(ACTIONS)) {
        register_module("prediction", network);
        network->to(ctx);
    }

    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
        return network->forward(x);
    }
};

struct Dynamics : torch::nn::Module {
    torch::Device ctx;
    std::shared_ptr<ResNet_dynamics<BasicBlock>> network;
    Dynamics(torch::Device& ctx) : ctx(ctx), network(resnet_dynamics(HIDDEN)) {
        register_module("dynamics", network);
        network->to(ctx);
    }

    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
        return network->forward(x);
    }
};

struct Network {
    at::TensorOptions ctx;
    std::shared_ptr<Representation> representation;
    std::shared_ptr<Dynamics> dynamics;
    std::shared_ptr<Prediction> prediction;

    int training_steps;

    Network(torch::Device ctx=get_ctx(), bool train=false) : ctx(ctx), representation(std::make_shared<Representation>(ctx)),
        dynamics(std::make_shared<Dynamics>(ctx)), prediction(std::make_shared<Prediction>(ctx)), training_steps(0) {
        representation->train(train);
        dynamics->train(train);
        prediction->train(train);
    }

    NetworkOutput initial_inference(Image_t &image) {
        torch::Tensor t = torch::tensor(image);
        torch::Tensor hidden_state_tensor = representation->forward(t);
        std::pair<torch::Tensor, torch::Tensor> prediction_output = prediction->forward(hidden_state_tensor);
        float * p = hidden_state_tensor.data<float>();
        float * l = prediction_output.first.data<float>();
        float * v = prediction_output.second.data<float>();
//        std::cout << prediction_output.first.size(0) << " " << prediction_output.first.size(1) << std::endl;
//        std::cout << hidden_state_tensor.size(0) << " " << hidden_state_tensor.size(1) << std::endl;
        return NetworkOutput(*v, 0, Policy_t(l, l + prediction_output.first.size(1)),
                HiddenState_t(p, p + hidden_state_tensor.size(1)),
                             prediction_output.second, torch::tensor({(float)0}).reshape({-1,1})
                             .to(ctx), prediction_output.first, hidden_state_tensor);
    }

    NetworkOutput recurrent_inference(HiddenState_t hidden_state, Action action) {
        std::pair<torch::Tensor, torch::Tensor> hidden_state_tensor = dynamics->forward(torch::tensor(hidden_state));
        std::pair<torch::Tensor, torch::Tensor>  prediction_output = prediction->forward(hidden_state_tensor.first);
        float * p = hidden_state_tensor.first.data<float>();
        float * r = hidden_state_tensor.second.data<float>();
        float * l = prediction_output.first.data<float>();
        float * v = prediction_output.second.data<float>();
//        std::cout << hidden_state_tensor.size(0) << " " << hidden_state_tensor.size(1) << std::endl;
        return NetworkOutput(*v, *r, Policy_t(l, l + prediction_output.first.size(1)),
                HiddenState_t(p, p + hidden_state_tensor.first.size(1)),
                prediction_output.second, hidden_state_tensor.second, prediction_output.first, hidden_state_tensor.first);
    }

    void zero_grad() {
        representation->zero_grad();
        dynamics->zero_grad();
        prediction->zero_grad();
    }


    void _save(std::string model_path, std::shared_ptr<torch::nn::Module> module) {
//        torch::save(module, model_path);

        torch::save(module, model_path);

//        torch::serialize::OutputArchive output_archive;
//        module.save(output_archive);
//        output_archive.save_to(model_path);

//        auto cu = std::make_shared<torch::jit::script::CompilationUnit>();
//        torch::serialize::OutputArchive archive(cu);
//        {
//            torch::serialize::OutputArchive slot(cu);
//            module.save(slot);
//            archive.write(module.name(), slot);
//        }
//        archive.save_to(model_path);
    }

    void save_network(std::string filename) {
        _save(filename + ".r.ckpt", representation);
        _save(filename + ".d.ckpt", dynamics);
        _save(filename + ".p.ckpt", prediction);
    }

    void _load(std::string model_path, std::shared_ptr<torch::nn::Module> module) {
//        std::ifstream in(model_path, std::ios::in | std::ios::binary);
//        torch::jit::script::Module m = torch::jit::load(in);
//        in.close();
//        return m;
//        torch::serialize::InputArchive archive;
//        archive.load_from(model_path);
//        module.load(archive);
        torch::load(module, model_path);
    }

    void load_network(std::string filename) {
        _load(filename + ".r.ckpt", representation);
        _load(filename + ".d.ckpt", dynamics);
        _load(filename + ".p.ckpt", prediction);
    }
};

struct SharedStorage {
    Network latest_network() {
        if (boost::filesystem::is_empty("/home/tomas/CLionProjects/muzero/network")) {
            return make_uniform_network();
        }
        Network network = make_uniform_network();
        network.load_network("/home/tomas/CLionProjects/muzero/network/latest");
        return network;
    }

    void save_network(int step, Network& network) {
        network.save_network("/home/tomas/CLionProjects/muzero/network/latest");
    }

    Network make_uniform_network() {
        return Network();
    }
};

int softmax_sample(std::vector<float> visit_counts, float temperature) {
//    std::cout << "t " << temperature << std::endl;
    std::vector<float> m;
    float mx = *std::max_element(visit_counts.begin(), visit_counts.end());
    for(int i = 0;i < visit_counts.size(); i++) {
        m.emplace_back(std::exp(visit_counts[i] - mx));
    }
    float counts_sum = std::accumulate(m.begin(), m.end(), (float)0.,
            [temperature](float &a, float &b){return a + b;});
    std::vector<float> d;
    for(int i = 0;i < visit_counts.size(); i++) {
        float s = m[i] / counts_sum;
//        std::cout << s << " / " << counts_sum << " / " << visit_counts[i] << " / " << m[i] << " / " << mx << std::endl;
        d.emplace_back(s);
    }

    std::default_random_engine generator;
    std::discrete_distribution<int> distribution(d.begin(), d.end());

    return distribution(generator);
}

void add_exploration_noise(MuZeroConfig& config, std::shared_ptr<Node>& node) {
    auto n = rnd.Dirichlet<ACTIONS>(config.root_dirichlet_alpha);
    float frac = config.root_exploration_fraction;
    for(int i = 0;i < node->children.size(); i++) {
        node->children[i]->prior = node->children[i]->prior * (1-frac) + n[i] * frac;
    }
}

void backpropagate(std::vector<std::shared_ptr<Node>> &search_path, float value, Player to_play, float discount,
        MinMaxStats& min_max_stats) {
    for(int i = 0; i < search_path.size(); i++) {
        std::shared_ptr<Node> node(search_path[i]);
        node->value_sum += node->to_play == to_play.id ? value : -value;
        node->visit_count +=1;
        min_max_stats.update(node->value());
        value = node->reward + discount * value;
    }
}

void expand_node(std::shared_ptr<Node>& node, Player to_play, ActionList_t actions, NetworkOutput network_output) {
    node->to_play = to_play.id;
    node->hidden_state = network_output.hidden_state;
    node->reward = network_output.reward;
    float policy_sum = std::accumulate(network_output.policy_logits.begin(), network_output.policy_logits.end(), (float)0,
            [](float &a, float &b ){ return a + std::exp(b) ;});
    for(int i = 0;i < network_output.policy_logits.size(); i++) {
        node->children.emplace_back(std::make_shared<Node>(network_output.policy_logits[i]/policy_sum));
    }
}


float ucb_score(MuZeroConfig& config, std::shared_ptr<Node> parent, std::shared_ptr<Node> child, MinMaxStats& min_max_stats) {
    float pb_c = std::log(((float)parent->visit_count + config.pb_c_base + 1) /
            config.pb_c_base) + config.pb_c_init;
    pb_c *= std::sqrt((float)parent->visit_count) / ((float)child->visit_count + 1);

    float prior_score = pb_c * child->prior;
    float value_score = min_max_stats.normalize(child->value());
    return prior_score + value_score;
}

std::pair<Action, std::shared_ptr<Node>> select_child(MuZeroConfig& config, std::shared_ptr<Node> node,
        MinMaxStats& min_max_stats) {
    float _ucb_score = -MAXIMUM_FLOAT_VALUE;
    int action = -1;
    std::shared_ptr<Node> child;
    for(int i = 0; i < node->children.size(); i++) {
        float u = ucb_score(config, node, node->children[i], min_max_stats);
        if (_ucb_score < u) {
            _ucb_score = u;
            action = i;
            child = node->children[i];
        }
    }

    assert(child != nullptr);
    return {Action(action), child};
}

Action select_action(MuZeroConfig& config, int num_moves, std::shared_ptr<Node>& node, Network& network) {
    std::vector<float> visit_counts;
    for(int i = 0;i < node->children.size(); i++) {
        visit_counts.emplace_back(node->children[i]->visit_count);
    }
    float t = config.visit_softmax_temperature_fn->operator()(num_moves, network.training_steps);
    int action = softmax_sample(visit_counts, t);
    return Action(action);

}

void run_mcts(MuZeroConfig& config, std::shared_ptr<Node>& root, ActionHistory action_history, Network& network) {
    MinMaxStats min_max_stats(config.known_bounds);


    for(int sim=0; sim < config.num_simulations; sim++) {
        ActionHistory history = action_history.clone();
        std::shared_ptr<Node> node(root);
        std::vector<std::shared_ptr<Node>> search_path;
        search_path.emplace_back(node);

        while(node->expanded()) {
            std::pair<Action, std::shared_ptr<Node>> r = select_child(config, node, min_max_stats);
            history.add_action(r.first);
            search_path.emplace_back(r.second);
            node = r.second;
        }

        std::shared_ptr<Node> parent = search_path[search_path.size() - 2];
        NetworkOutput network_output = network.recurrent_inference(parent->hidden_state, history.last_action());
        expand_node(node, history.to_play(), history.action_space(), network_output);

        backpropagate(search_path, network_output.value, history.to_play(), config.discount, min_max_stats);

    }
}

std::shared_ptr<Game> play_game(MuZeroConfig& config, Network& network) {
    std::shared_ptr<Game> game = config.new_game();

    int count = 0;
    while(!game->terminal() && game->history.size() < config.max_moves) {
        std::shared_ptr<Node> root(std::make_shared<Node>(0));
        Image_t current_observation = game->make_image(-1);
        expand_node(root, game->to_play(), game->legal_actions(),
                network.initial_inference(current_observation));
        add_exploration_noise(config, root);

        std::cout << "run_mcts " << count << std::endl;
        count++;
        run_mcts(config, root, game->action_history(), network);
        Action action = select_action(config, game->history.size(), root, network);
        game->apply(action);
        game->store_search_statistics(root);
    }

    return game;
}

void run_selfplay(MuZeroConfig config, SharedStorage storage, ReplayBuffer replay_buffer, int tid) {
    for(;;) {
        Network network = storage.latest_network();
        std::shared_ptr<Game> game = play_game(config, network);
        replay_buffer.save_game(game);
    }
}

torch::Tensor cross_entropy_loss(torch::Tensor input, torch::Tensor target) {
//    return torch::nll_loss(torch::log_softmax(input, 1), target, {}, Reduction::Sum, -100);
//    std::cout << -torch::log_softmax(input,1) << std::endl;
//    std::cout << input.size(0) << input.size(1) << std::endl;
//    std::cout << target.size(0)  << std::endl;
    return -(target.reshape({-1, ACTIONS}) * torch::log_softmax(input,1)).sum(1).mean();
}

void update_weights(torch::optim::Optimizer& opt, Network& network, std::vector<Batch> batch, float weight_decay, torch::Device ctx) {

    for(int i = 0; i < batch.size(); i++) {
        std::vector<NetworkOutput> predictions;
        network.zero_grad();
        Image_t image = batch[i].image;
        ActionList_t actions = batch[i].action;
        std::vector<Target> targets = batch[i].target;
        NetworkOutput network_output = network.initial_inference(image);

        predictions.emplace_back(network_output);

        for(int j = 0; j < actions.size(); j++) {
            NetworkOutput network_output_1 = network.recurrent_inference(network_output.hidden_state, actions[j]);
            predictions.emplace_back(network_output_1);
        }

        torch::Tensor values_v;
        std::vector<float> target_values_v;
        torch::Tensor rewards_v;
        std::vector<float> target_rewards_v;
        torch::Tensor logits_v;
        torch::Tensor target_policies_v;

        for(int k = 0; k < std::min(predictions.size()-1, targets.size()); k++) {
            if(k == 0) {
                values_v = predictions[k].value_tensor;
                rewards_v = predictions[k].reward_tensor;
                logits_v = predictions[k].policy_tensor;
                target_policies_v = torch::tensor(targets[k].policy);
            } else {
                values_v = torch::cat({values_v, predictions[k].value_tensor});
//                std::cout << rewards_v.size(0) << " / " << rewards_v.size(0) << std::endl;
//                std::cout << predictions[k].reward_tensor.size(0) << " / " << predictions[k].reward_tensor.size(1) << std::endl;
                rewards_v = torch::cat({rewards_v, predictions[k].reward_tensor});
                logits_v = torch::cat({logits_v, predictions[k].policy_tensor});
                target_policies_v = torch::cat({target_policies_v, torch::tensor(targets[k].policy)});
            }
            target_values_v.emplace_back(targets[k].value);
            target_rewards_v.emplace_back(targets[k].reward);
//            target_policies_v.emplace_back(targets[k].policy);

        }

        torch::Tensor values = values_v.to(ctx);
        torch::Tensor target_values = torch::tensor(target_values_v).reshape({-1,1}).to(ctx);
        torch::Tensor rewards = rewards_v.to(ctx);
        torch::Tensor target_rewards = torch::tensor(target_rewards_v).reshape({-1,1}).to(ctx);
        torch::Tensor logits = logits_v.reshape({-1, ACTIONS}).to(ctx);
//        std::cout << logits.size(0) << " / " << logits.size(1) << std::endl;
//        torch::Tensor target_policies = torch::from_blob(target_policies_v.data(),
//                {static_cast<long>(target_policies_v.size()), static_cast<long>(target_policies_v[0].size())}).to(ctx);
//        std::cout << target_policies.size(0) << " / " << target_policies.size(1) << std::endl;

//        std::cout << "values: " <<  values << std::endl;
//        std::cout << "target_values: " << target_values << std::endl;
//        std::cout << "logits: " << logits << std::endl;
//        std::cout << "target_policies: " << target_policies << std::endl;
        torch::Tensor l = torch::mse_loss(values, target_values) + torch::mse_loss(rewards, target_rewards)
                + cross_entropy_loss(logits, target_policies_v.to(ctx));
//                + torch::poisson_nll_loss(logits, target_policies, false, true, 1e-8, Reduction::Mean);
        if(i == 0) {
            std::cout << "\t\t loss: " << l << std::endl;
        }

        float abc = *((float*)l.data<float>());
        assert(!isnan(abc));
        l.backward({}, false, false);
        opt.step();

    }
}

void train_network(MuZeroConfig& config, SharedStorage& storage, ReplayBuffer& replay_buffer, torch::Device ctx) {
    Network network(ctx,true);
    std::vector<torch::Tensor> params({});
    std::vector<torch::Tensor> r_params = network.representation->parameters();
    std::vector<torch::Tensor> d_params = network.dynamics->parameters();
    std::vector<torch::Tensor> p_params = network.prediction->parameters();

    std::copy(r_params.begin(), r_params.end(), std::back_inserter(params));
    std::copy(d_params.begin(), d_params.end(), std::back_inserter(params));
    std::copy(p_params.begin(), p_params.end(), std::back_inserter(params));

    torch::optim::Adam opt(params, torch::optim::AdamOptions(config.lr_init));

    for(int i = 0; i < config.training_steps; i++) {
        std::cout << "\t train step: " << i << std::endl;
        if(i != 0 && i % config.checkpoint_interval == 0) {
            storage.save_network(i, network);
        }
        std::vector<Batch> batch = replay_buffer.sample_batch(config.num_unroll_steps, config.td_steps);
        update_weights(opt, network, batch, config.weight_decay, ctx);
        network.training_steps++;
    }
    storage.save_network(config.training_steps, network);

}


Network muzero(MuZeroConfig config) {
    SharedStorage storage;
    ReplayBuffer replay_buffer(config);

    std::vector<std::shared_ptr<std::thread>> threads;
    for(int i = 0; i < config.num_actors; i++) {
//        run_selfplay(config, storage, replay_buffer, i);
        threads.emplace_back(std::make_shared<std::thread>(run_selfplay, config, storage, replay_buffer, i));
//        th.join();
    }

    train_network(config, storage, replay_buffer, get_train_ctx());


    return storage.latest_network();
}




int main(int argc, char** argv) {
    muzero(make_c_config());
    return 0;
}