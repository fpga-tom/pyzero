/**
                    GNU GENERAL PUBLIC LICENSE
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

#include "main.h"

torch::Device get_ctx() {
    return torch::Device("cuda:0");
}

torch::Device get_train_ctx() {
    return torch::Device("cuda:0");
}

torch::Device get_cpu_ctx() {
    return torch::Device("cpu:0");
}



Random rnd(Random::kUniqueSeed, Random::kUniqueStream);
std::random_device seeder;
std::default_random_engine engine(seeder());
std::default_random_engine generator;

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
            KnownBounds known_bounds,
            bool train
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
            checkpoint_interval(int(500)),
            window_size(int(1e6)),
            batch_size(batch_size),
            num_unroll_steps(5),
            td_steps(td_steps),

            weight_decay(1e-4),
            momentum(0.9),

            // Exponential learning rate schedule
            lr_init(lr_init),
            lr_decay_rate(0.1),
            lr_decay_steps(lr_decay_steps),
            train(train),
            num_selfplay(int(1000e3)),
            num_executors(4) {

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
    bool train;
    std::string path;
    int num_selfplay;
    int num_executors;

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
            if (num_moves < 2)
                return 1.0;
            else
                return 0.0;
        }
    };

    return {
            action_space_size,
            max_moves, 1.0,
            dirichlet_alpha,
            93,
            128,
            max_moves,  //Always use Monte Carlo return.
            1,
            lr_init,
            400e3,
            new VisitSoftmaxTemperatureFn1(),
            KnownBounds(-1, 1),
            false
    };
}

MuZeroConfig make_c_config() {
    return make_board_config(ACTIONS, 6, 0.03, 0.001);
}



struct Player {
    int id;

    Player(int id=1) : id(id) {

    }

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
        assert(!isnan(prior));

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

    ActionHistory(ActionList_t history, int action_space_size) :
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

    Player to_play() {
        if(history.size() % 2 == 0)
            return 1;
        else
            return -1;

    }
};

template <typename T>
std::ofstream& write(std::ofstream &out, const T& obj) {
    out.write(reinterpret_cast<const char *>(&obj), sizeof(T));
    return out;
}

template <typename T>
std::ifstream& read(std::ifstream &in, T& obj) {
    in.read(reinterpret_cast<char *>(&obj), sizeof(T));
    return in;
}

template <typename T>
std::ofstream &write (std::ofstream &out, const std::vector<T>& obj) {
    typename std::vector<T>::size_type count = obj.size();
    out.write(reinterpret_cast<const char *>(&count), sizeof(count));
    out.write(reinterpret_cast<const char *>(obj.data()), sizeof(T) * obj.size());
    return out;
}

template <typename T>
std::ofstream &write (std::ofstream &out, const std::vector<std::vector<T>>& obj) {
    typename std::vector<T>::size_type count = obj.size();
    out.write(reinterpret_cast<const char *>(&count), sizeof(count));
    for(typename std::vector<T>::size_type i = 0; i < obj.size(); i++)
        write(out, obj[i]);
    return out;
}

template <typename T>
std::ifstream &read (std::ifstream &in, std::vector<T>& obj) {
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

std::ifstream &read_action (std::ifstream &in, ActionList_t & obj) {
    ActionList_t::size_type count=0;
    in.read(reinterpret_cast<char *>(&count), sizeof(count));

    for(int i = 0; i < count; i++) {
        Action a;
        read(in,a);
        obj.emplace_back(a);
    }
    return in;
}

std::ofstream &write_action (std::ofstream &out, const ActionList_t & obj) {
    ActionList_t::size_type count=obj.size();
    out.write(reinterpret_cast<char *>(&count), sizeof(count));

    for(ActionList_t::size_type i = 0; i < count; i++) {
        write(out,obj[i]);
    }
    return out;
}

template <typename T>
std::ifstream &read (std::ifstream &in, std::vector<std::vector<T>>& obj) {
    typename std::vector<T>::size_type count = 0;
    in.read(reinterpret_cast<char *>(&count), sizeof(count));
    for(typename std::vector<T>::size_type i = 0; i < count; i++) {
        std::vector<T> obj1;
        read(in, obj1);
        obj.emplace_back(obj1);
    }
    return in;
}

struct Environment {

    std::ifstream f;
    std::vector<long> seq;
    std::vector<int> rewards;

    Environment() : seq({}) {
        f.open("/home/tomas/CLionProjects/muzero/example.txt");
        if (!f) {
            std::cerr << "unable to open example.txt" << std::endl;
        }
    }

    int h_dist(const std::string& a, const std::string& b, int p) {
        int count=0;
        for(int i=0; i<std::min(a.size(), b.size()); i++)
        {
            if( i % 2 == p)
                count += a[i] == b[i];
//            int partial = (~((a[i] & 0xFF) ^ (b[i] & 0xFF)))&0xFF;
//            while(partial)
//            {
//                count += partial & 1;
//                partial = partial >>1;
//            }
        }
        return count;
    }

    void print_state (const std::ios& stream) {
        std::cout << " good()=" << stream.good();
        std::cout << " eof()=" << stream.eof();
        std::cout << " fail()=" << stream.fail();
        std::cout << " bad()=" << stream.bad();
    }

    float step(Action &action) {
        std::string line;
        seq.emplace_back(action.index);
        std::stringstream result;
        for(int i = 0; i < seq.size(); i++) {
            result << PRINTABLE[seq[i]];
        }
        std::string str = result.str();

        f.clear();
        f.seekg(0, std::ios::beg);
        assert(f.good());
        int d_0 = 0;
        int d_1 = 0;
        if (seq.size() < 6) {
            return 0;
        }
        std::cout << str << std::endl;
        while (f.good()) {
            getline(f, line);
            if(line.size() >= str.size()) {
                for (size_t i = 0; i <= line.size() - str.size(); i++) {
                    int d_0_ = h_dist(str, line.substr(i), 0);
                    int d_1_ = h_dist(str, line.substr(i), 1);

                    int d = std::max(d_0, d_1);
                    int d_ = std::max(d_0_, d_1_);

                    if(d_ > d) {
                        d_0 = d_0_;
                        d_1 = d_1_;
                    }
                }
            }
        }

        std::cout << d_0 << " " << d_1 << std::endl;
        if(d_0 > d_1) {
            return -1;
        } else if(d_0 < d_1) {
            return 1;
        }

        return 0;

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

    friend std::ofstream &write (std::ofstream &out, const Environment& obj) {
        write(out, obj.seq);
        write(out, obj.rewards);
        return out;
    }

    friend std::ifstream &read (std::ifstream &in, Environment& obj) {
        read(in, obj.seq);
        read(in, obj.rewards);
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
        /*
        if(history.size() > 0) {
            return rewards[rewards.size()-1] == -1;
        }
        return false;
         */
        return history.size() == 6;
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
        state_index = state_index < 0 ? (int)environment.seq.size() + state_index + 1: state_index;
        int high = state_index;
        int low = std::max(high - HISTORY, 0);
        std::copy(environment.seq.begin() + low, environment.seq.begin() + high , std::back_inserter(r));
        assert(r.size() == high);
        int pad = HISTORY - r.size();
        for(int i = 0;i < pad; i++) {
            r.emplace_back(ACTIONS);
        }
        return r;
    }

    std::vector<Target> make_target(int state_index, int num_unroll_steps, int td_steps, Player to_play) {
        assert(root_values.size() == history.size());
        std::vector<Target> targets;
        for(int current_index = state_index; current_index < state_index + num_unroll_steps + 1; current_index++) {
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
//                assert(child_visits[current_index].size() > 0);
//                for(int i = 0; i < child_visits[current_index].size(); i++) {
//                    assert(!isnan(child_visits[current_index][i]));
//                }
                targets.emplace_back(Target(value, rewards[current_index], child_visits[current_index]));
            } else {
                targets.emplace_back(Target(0, 0, {}));
            }
        }
        return targets;
    }


    Player to_play() {
        if(history.size() % 2 == 0)
            return 1;
        else
            return -1;
    }

    friend std::ofstream &write (std::ofstream &out, const Game& obj) {
        write(out , obj.environment);
        write_action(out , obj.history);
        write(out , obj.rewards);
        write(out , obj.child_visits);
        write(out , obj.root_values);
        write(out , obj.action_space_size);
        write(out , obj.discount);
        return out;
    }

    friend std::ifstream &read (std::ifstream &in, Game& obj) {
        read(in , obj.environment);
        read_action(in , obj.history);
        read(in , obj.rewards);
        read(in , obj.child_visits);
        read(in , obj.root_values);
        read(in , obj.action_space_size);
        read(in , obj.discount);
        return in;
    }

    void save(const std::string& filename) {
        std::string path = "/home/tomas/CLionProjects/muzero/replay_buffer/" + filename;
        assert(!boost::filesystem::is_regular_file(path));
        std::ofstream out(path, std::ios::out | std::ios::binary);
        assert(history.size() > 0);
        assert(child_visits.size() > 0);
        assert(child_visits[0].size() > 0);
        assert(action_space_size == ACTIONS);
        assert(discount == (float)1.);
        for(int i = 0; i < child_visits.size(); i++) {
            for(int j = 0; j < child_visits[i].size(); j++)
                assert(!isnan(child_visits[i][j]));
        }
        write(out , *this);
        out.close();
        assert(boost::filesystem::is_regular_file(path));
    }

    void load(const std::string& filename) {
        assert(boost::filesystem::is_regular_file(filename));
        std::ifstream in(filename, std::ios::in | std::ios::binary);
        read(in, *this);
        in.close();
        assert(environment.seq.size() > 0);
        assert(history.size() > 0);
        assert(child_visits.size() > 0);
        assert(child_visits[0].size() > 0);
        assert(action_space_size == ACTIONS);
        assert(discount == 1.);


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
    int batch_idx;
    std::vector<std::string> game_files;

    ReplayBuffer(MuZeroConfig& config) : window_size(config.window_size),
        batch_size(config.batch_size), batch_idx(0),
        buffer({}), game_files({}) {
    }

    void save_game(std::shared_ptr<Game> game) {
//        if(buffer.size() > window_size) {
//            buffer.erase(buffer.begin());
//        }
//        buffer.emplace_back(game);
        boost::uuids::random_generator gen;
        boost::uuids::uuid u = gen();
        game->save(boost::uuids::to_string(u));
    }

    void shuffle() {
        result_set_t result_set;
        std::string path = "/home/tomas/CLionProjects/muzero/replay_buffer/";
        game_files.clear();
        batch_idx = 0;

        while(!boost::filesystem::is_directory(path)) {
            sleep(5);
        }


        while (boost::filesystem::is_empty(path)) {
            sleep(5);
        }
        for(boost::filesystem::directory_iterator it(path); it != boost::filesystem::directory_iterator(); ++it) {
            if (boost::filesystem::is_regular_file(it->status()) )
            {
                result_set.insert(result_set_t::value_type(boost::filesystem::last_write_time(it->path()), *it));
            }
        }

        for(auto it = result_set.begin(); it != result_set.end(); ++it) {
            game_files.emplace_back((*it).second.c_str());
        }

        if(game_files.size() > window_size) {
            game_files.erase(game_files.begin(), game_files.end() - window_size);
        }

        std::random_shuffle(game_files.begin(), game_files.end());
    }

    std::vector<Batch> sample_batch(int num_unroll_steps, int td_steps) {
        std::vector<Batch> result;
        std::vector<std::shared_ptr<Game>> games;

        for(int i = 0; i < batch_size; i++) {
            games.emplace_back(sample_game(game_files));
        }

        std::vector<int> game_pos;
        for(int i = 0; i < games.size(); i++) {
            game_pos.emplace_back(sample_position(games[i]));
        }
        for(int i = 0; i < game_pos.size(); i++) {
            auto g = games[i];
            Image_t image = g->make_image(game_pos[i]);
            ActionList_t history;
            std::copy(g->history.begin()+game_pos[i], g->history.begin()+std::min(game_pos[i]+num_unroll_steps, (int)g->history.size()) - 1,
                    std::back_inserter(history));
            std::vector<Target> target = g->make_target(game_pos[i], num_unroll_steps, td_steps, g->to_play());
            result.emplace_back(Batch{image, history, target});
        }
        return result;
    }

    std::shared_ptr<Game> sample_game(std::vector<std::string>& result_set) {


        auto game = std::make_shared<Game>(0, 0);
        if(batch_idx >= result_set.size()) {

            if(game_files.empty()) {
                shuffle();
            } else {
                batch_idx = 0;
                std::random_shuffle(game_files.begin(), game_files.end());
            }


        }
        game->load(result_set[batch_idx]);

        batch_idx++;

        assert(game->action_space_size == ACTIONS);
        assert(game->discount == (float)1.);
        return game;

    }

    int sample_position(std::shared_ptr<Game>& game) {

        std::uniform_int_distribution<int> dist(0, game->history.size() - 1);
        int guess = dist(engine);
        return guess;
    }
};


torch::nn::Conv1dOptions conv_options(int64_t in_planes, int64_t out_planes, int64_t kerner_size,
                                      int64_t stride=1, int64_t padding=0, bool with_bias=false){
    torch::nn::Conv1dOptions conv_options = torch::nn::Conv1dOptions(in_planes, out_planes, kerner_size);
    conv_options.stride(stride);
    conv_options.padding(padding);
    conv_options.with_bias(with_bias);
    return conv_options;
}


struct BasicBlock : torch::nn::Module {

    static const int expansion;

    int64_t stride;
    torch::nn::Conv1d conv1;
    torch::nn::BatchNorm bn1;
    torch::nn::Conv1d conv2;
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
    torch::nn::Conv1d conv1;
    torch::nn::BatchNorm bn1;
    torch::nn::Conv1d conv2;
    torch::nn::BatchNorm bn2;
    torch::nn::Conv1d conv3;
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
        at::Tensor residual= x;

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

    int64_t inplanes = 32;
    torch::nn::Conv1d conv1;
    torch::nn::BatchNorm bn1;
//    torch::nn::Sequential layer1;
//    torch::nn::Sequential layer2;
//    torch::nn::Sequential layer3;
//    torch::nn::Sequential layer4;
    torch::nn::Linear fc;
    torch::nn::Embedding embedding;

    ResNet_representation(torch::IntList layers, int64_t num_classes=1000)
            :
            conv1(conv_options(HISTORY, 32, 3, 2, 1)),
              bn1(32),
//              layer1(_make_layer(32, layers[0])),
//              layer2(_make_layer(64, layers[1], 2)),
//              layer3(_make_layer(64, layers[2], 2)),
//              layer4(_make_layer(64, layers[3], 2)),
              fc(256 * Block::expansion, num_classes),
              embedding(torch::nn::Embedding(ACTIONS+1, ACTIONS + 1))
    {
        register_module("conv1", conv1);
        register_module("bn1", bn1);
//        register_module("layer1", layer1);
//        register_module("layer2", layer2);
//        register_module("layer3", layer3);
//        register_module("layer4", layer4);
        register_module("fc", fc);
        register_module("embedding", embedding);
//        init_x(conv1->parameters());


        torch::Tensor t;
        for(int i = 0; i < ACTIONS+1; i++) {
            std::vector<float> v;
            for(int j = 0; j < ACTIONS + 1; j++) {
                if(j == i) {
                    v.emplace_back(1);
                } else {
                    v.emplace_back(0);
                }
            }
            if (i == 0) {
                t = torch::tensor(v).reshape({-1, ACTIONS + 1});
            } else {
                t = torch::cat({t, torch::tensor(v).reshape({-1, ACTIONS + 1})});
            }
        }
//        embedding->weight=t.to(get_ctx()).set_requires_grad(false);
        // Initializing weights
        for(auto m: modules(false)){
            if (m->name() == "torch::nn::Conv1dImpl"){
                for (auto p: m->parameters()){
//                    torch::nn::init::xavier_normal_(p);
                    torch::nn::init::kaiming_uniform_(p, 0, torch::nn::init::FanMode::FanIn,
                            torch::nn::init::Nonlinearity::ReLU);
                }
            }
            if (m->name() == "torch::nn::LinearImpl"){
                for (auto p: m->named_parameters()){
                    if (p.key() == "weight"){
//                        torch::nn::init::xavier_normal_(*p);
                        torch::nn::init::kaiming_uniform_(*p, 0, torch::nn::init::FanMode::FanIn,
                                torch::nn::init::Nonlinearity::ReLU);
                    }
                    else if (p.key() == "bias"){
                        torch::nn::init::constant_(*p, 0);
//                        torch::nn::init::xavier_normal_(*p);
                    }
                }
            }
            if (m->name() == "torch::nn::EmbeddingImpl"){
                for (auto p: m->parameters()){
//                    torch::nn::init::xavier_normal_(p);
                    torch::nn::init::kaiming_uniform_(p, 0, torch::nn::init::FanMode::FanIn,
                            torch::nn::init::Nonlinearity::ReLU);
                }
            }
            if (m->name() == "torch::nn::BatchNormImpl"){
                for (auto p: m->named_parameters()){
                    if (p.key() == "weight"){
                        torch::nn::init::constant_(*p,1);
                    }
                    else if (p.key() == "bias"){
                        torch::nn::init::constant_(*p, 0);
                    }
                }
            }
        }
    }


    torch::Tensor forward(torch::Tensor x){

        torch::Tensor emb = embedding(x);
        x = emb.view({-1,HISTORY, ACTIONS + 1});
        x = conv1->forward(x);
        x = bn1->forward(x);
        x = torch::relu(x);
        x = torch::avg_pool1d(x, 3, 2, 1);

//        x = layer1->forward(x);
//        x = layer2->forward(x);
//        x = layer3->forward(x);
//        x = layer4->forward(x);

//        x = torch::avg_pool1d(x, 3, 3, 1);
        x = x.view({x.sizes()[0], -1});
        x = fc->forward(x);
        x = torch::tanh(x);

        return x;
    }


private:
    torch::nn::Sequential _make_layer(int64_t planes, int64_t blocks, int64_t stride=1){
        torch::nn::Sequential downsample;
        if (stride != 1 or inplanes != planes * Block::expansion){
            downsample = torch::nn::Sequential(
                    torch::nn::Conv1d(conv_options(inplanes, planes * Block::expansion, 1, stride)),
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
//    torch::nn::Conv1d conv1;
//    torch::nn::BatchNorm bn1;
//    torch::nn::Sequential layer1;
//    torch::nn::Sequential layer2;
//    torch::nn::Sequential layer3;
//    torch::nn::Sequential layer4;
//    torch::nn::Linear fc;
//    torch::nn::Linear fc1;
    torch::nn::Embedding embedding;

    torch::nn::Conv1d conv2;
    torch::nn::BatchNorm bn2;
    torch::nn::Linear fc1;
    torch::nn::Linear fc2;

    torch::nn::Conv1d conv3;
    torch::nn::BatchNorm bn3;
    torch::nn::Linear fc3;
    torch::nn::Linear fc4;

    ResNet_dynamics(torch::IntList layers, int64_t num_classes=1000)
            :
//            conv1(conv_options(1, 64, 3, 1, 1)),
//              bn1(64),
//              layer1(_make_layer(64, layers[0], 1)),
//              layer2(_make_layer(128, layers[1], 2)),
//              layer3(_make_layer(128, layers[2], 2)),
//              layer4(_make_layer(128, layers[3], 2)),
//              fc(512 * Block::expansion, num_classes),
//              fc1(512 * Block::expansion, 1),
              conv2(conv_options(1 , 16,3,2,1)),
              bn2(16),
              fc1(256 * Block::expansion, 64),
              fc2(64, 1),
              conv3(conv_options(1 , 16,3,2,1)),
              bn3(16),
              fc3(256 * Block::expansion, 64),
              fc4(64, num_classes),
              embedding(torch::nn::Embedding(ACTIONS+1, ACTIONS+1))
    {
//        register_module("conv1", conv1);
//        register_module("bn1", bn1);
//        register_module("layer1", layer1);
//        register_module("layer2", layer2);
//        register_module("layer3", layer3);
//        register_module("layer4", layer4);
//        register_module("fc", fc);
//        register_module("fc1", fc1);
        register_module("embedding", embedding);

        register_module("conv2", conv2);
        register_module("bn2", bn2);
        register_module("fc1", fc1);
        register_module("fc2", fc2);

        register_module("conv3", conv3);
        register_module("bn3", bn3);
        register_module("fc3", fc3);
        register_module("fc4", fc4);

        torch::Tensor t;
        for(int i = 0; i < ACTIONS+1; i++) {
            std::vector<float> v;
            for(int j = 0; j < ACTIONS + 1; j++) {
                if(j == i) {
                    v.emplace_back(1);
                } else {
                    v.emplace_back(0);
                }
            }
            if (i == 0) {
                t = torch::tensor(v).reshape({-1, ACTIONS + 1});
            } else {
                t = torch::cat({t, torch::tensor(v).reshape({-1, ACTIONS + 1})});
            }
        }
//        embedding->weight=t.to(get_ctx()).set_requires_grad(false);

        // Initializing weights
        for(auto m: this->modules(false)){
            if (m->name() == "torch::nn::Conv1dImpl"){
                for (auto p: m->parameters()){
//                    torch::nn::init::xavier_normal_(p);
                    torch::nn::init::kaiming_uniform_(p, 0, torch::nn::init::FanMode::FanIn,
                            torch::nn::init::Nonlinearity::ReLU);
                }
            }
            if (m->name() == "torch::nn::LinearImpl"){
                for (auto p: m->named_parameters()){
                    if (p.key() == "weight"){
//                        torch::nn::init::xavier_normal_(*p);
                        torch::nn::init::kaiming_uniform_(*p, 0, torch::nn::init::FanMode::FanIn,
                                torch::nn::init::Nonlinearity::ReLU);
                    }
                    else if (p.key() == "bias"){
                        torch::nn::init::constant_(*p, 0);
//                        torch::nn::init::xavier_normal_(*p);
                    }
                }
            }
            if (m->name() == "torch::nn::EmbeddingImpl"){
                for (auto p: m->parameters()){
//                    torch::nn::init::xavier_normal_(p);
                    torch::nn::init::kaiming_uniform_(p, 0, torch::nn::init::FanMode::FanIn,
                            torch::nn::init::Nonlinearity::ReLU);
                }
            }
            if (m->name() == "torch::nn::BatchNormImpl"){
                for (auto p: m->named_parameters()){
                    if (p.key() == "weight"){
                        torch::nn::init::constant_(*p, 1);
                    }
                    else if (p.key() == "bias"){
                        torch::nn::init::constant_(*p, 0);
                    }
                }
            }
        }
    }

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x, torch::Tensor action){

        action = embedding->forward(action);
        x = torch::cat({x.view({-1, HIDDEN}), action.view({-1, ACTIONS + 1})}, 1);
        x = x.view({-1,1, HIDDEN + ACTIONS + 1});
//        x = conv1->forward();
//        x = bn1->forward(x);
//        x = torch::relu(x);
//        x = torch::max_pool1d(x, 3, 1, 1);
//
//        x = layer1->forward(x);
//        x = layer2->forward(x);
//        x = layer3->forward(x);
//        x = layer4->forward(x);
//
//        x = torch::avg_pool1d(x, 3, 1, 1);
//        torch::Tensor y = x.view({x.sizes()[0], -1});
//        y = y.flatten();
//        x = fc->forward(y).sigmoid().reshape({-1, HIDDEN});
//        y = torch::tanh(fc1->forward(y)).reshape({-1, 1});


        torch::Tensor tmp = x;
//        torch::Tensor y = x.view({x.sizes()[0], -1});

//        x = fc->forward(y);
        torch::Tensor y = conv2->forward(tmp);
        y = bn2->forward(y);
        y = torch::relu(y);
        y = torch::max_pool1d(y, 7, 3, 2);
        y = fc1->forward(y.view({y.sizes()[0], 1, -1}));
        y = torch::relu(y);
        y = fc2->forward(y);
//        y = torch::tanh(y);
//        y = y.clamp(-1, 1);
//        y = torch::tanh(fc2->forward(y));

        x = conv3->forward(tmp);
        x = bn3->forward(x);
        x = torch::relu(x);
        x = torch::max_pool1d(x, 7, 3, 2);
        x = fc3->forward(x.view({x.sizes()[0], 1, -1}));
//        x = x.clamp(-1, 1);
        x = fc4->forward(torch::relu(x));
        x = torch::tanh(x);

        return {x,y};
    }


private:
    torch::nn::Sequential _make_layer(int64_t planes, int64_t blocks, int64_t stride=1){
        torch::nn::Sequential downsample;
        if (stride != 1 or inplanes != planes * Block::expansion){
            downsample = torch::nn::Sequential(
                    torch::nn::Conv1d(conv_options(inplanes, planes * Block::expansion, 1, stride)),
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
//    torch::nn::Conv1d conv1;
//    torch::nn::BatchNorm bn1;
//    torch::nn::Sequential layer1;
//    torch::nn::Sequential layer2;
//    torch::nn::Sequential layer3;
//    torch::nn::Sequential layer4;
//    torch::nn::Linear fc;

    torch::nn::Conv1d conv2;
    torch::nn::BatchNorm bn2;
    torch::nn::Linear fc1;
//    torch::nn::Linear fc2;

    torch::nn::Conv1d conv3;
    torch::nn::BatchNorm bn3;
    torch::nn::Linear fc3;
//    torch::nn::Linear fc4;

    ResNet_prediction(torch::IntList layers, int64_t num_classes=1000)
            :
//            conv1(conv_options(1, 32, 7, 3, 2)),
//              bn1(32),
//              layer1(_make_layer(64, layers[0])),
//              layer2(_make_layer(128, layers[1], 1)),
//              layer3(_make_layer(128, layers[2], 2)),
//              layer4(_make_layer(128, layers[3], 2)),
//              fc(1024 * Block::expansion, num_classes),
              conv2(conv_options(1, 8,5,3,2)),
              bn2(8),
              fc1(88 * Block::expansion, 1),
//              fc2(32, 1),
    conv3(conv_options(1, 8,5,3,2)),
    bn3(8),
    fc3(88 * Block::expansion, num_classes)
//    fc4(32, num_classes)
    {
//        register_module("conv1", conv1);
//        register_module("bn1", bn1);
//        register_module("layer1", layer1);
//        register_module("layer2", layer2);
//        register_module("layer3", layer3);
//        register_module("layer4", layer4);
//        register_module("fc", fc);
        register_module("conv2", conv2);
        register_module("bn2", bn2);
        register_module("fc1", fc1);
//        register_module("fc2", fc2);

        register_module("conv3", conv3);
        register_module("bn3", bn3);
        register_module("fc3", fc3);
//        register_module("fc4", fc4);

        // Initializing weights
        for(auto m: this->modules(false)){
            if (m->name() == "torch::nn::Conv1dImpl"){
                for (auto p: m->parameters()){
//                    torch::nn::init::xavier_normal_(p);
                    torch::nn::init::kaiming_uniform_(p, 0, torch::nn::init::FanMode::FanIn,
                            torch::nn::init::Nonlinearity::ReLU);
                }
            }
            if (m->name() == "torch::nn::LinearImpl"){
                for (auto p: m->named_parameters()){
                    if (p.key() == "weight"){
//                        torch::nn::init::xavier_normal_(*p);
                        torch::nn::init::kaiming_uniform_(*p, 0, torch::nn::init::FanMode::FanIn,
                                torch::nn::init::Nonlinearity::ReLU);
                    }
                    else if (p.key() == "bias"){
                        torch::nn::init::constant_(*p, 1);
//                    torch::nn::init::kaiming_uniform_(*p, 0, torch::nn::init::FanMode::FanIn, torch::nn::init::Nonlinearity::ReLU);
                    }
                }
            }
            if (m->name() == "torch::nn::BatchNormImpl"){
                for (auto p: m->named_parameters()){
                    if (p.key() == "weight"){
                        torch::nn::init::constant_(*p, 1);
                    }
                    else if (p.key() == "bias"){
                        torch::nn::init::constant_(*p, 0);
                    }
                }
            }
        }
    }

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x){

        x = x.view({-1, 1, HIDDEN});
        DUMP_LOG(x.sizes());
//        x = conv1->forward(x);
//        x = bn1->forward(x);
//        x = torch::relu(x);
//        DUMP_LOG(x.sizes());
//        x = torch::max_pool1d(x, 3, 2, 1);
//
//        x = layer1->forward(x);
//        DUMP_LOG(x.sizes());
//        x = layer2->forward(x);
//        DUMP_LOG(x.sizes());
//        x = layer3->forward(x);
        DUMP_LOG(x.sizes());
//        x = layer4->forward(x);
        DUMP_LOG(x.sizes());

//        x = torch::avg_pool1d(x, 3, 2, 1);
        torch::Tensor tmp = x;
        DUMP_LOG(x.sizes());
//        torch::Tensor y = x.view({x.sizes()[0], -1});

//        x = fc->forward(y);
        torch::Tensor y = conv2->forward(tmp);
//        DUMP_LOG(y.sizes());
        y = bn2->forward(y);

        y = torch::relu(y);
        y = torch::max_pool1d(y, 5, 2, 2);
        y = fc1->forward(y.view({y.sizes()[0], 1, -1}));
//        y = fc2->forward(torch::relu(y));
//        y = torch::relu(y);
        y = y.clamp(-10, 10);
//        val = torch::tanh(val);
        DUMP_LOG(y.sizes());
//        y = fc2->forward(y);

        x = conv3->forward(tmp);
        x = bn3->forward(x);
        x = torch::relu(x);
        x = torch::max_pool1d(x, 5, 2, 2);
        x = fc3->forward(x.view({x.sizes()[0], 1, -1}));
//        x = torch::relu(x);
//        x = fc4->forward(x);
        x = x.clamp(-10, 10);

        DUMP_LOG(x.sizes());
        DUMP_LOG(y.sizes());
        return {x, y};
    }


private:
    torch::nn::Sequential _make_layer(int64_t planes, int64_t blocks, int64_t stride=1){
        torch::nn::Sequential downsample;
        if (stride != 1 or inplanes != planes * Block::expansion){
            downsample = torch::nn::Sequential(
                    torch::nn::Conv1d(conv_options(inplanes, planes * Block::expansion, 1, stride)),
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

    Representation(torch::Device& ctx) : ctx(ctx), network(resnet_representation(HIDDEN))
         {
        register_module("representation", network);
    }

    torch::Tensor forward(torch::Tensor x) {
        return network->forward(x);
    }
};

struct Prediction : torch::nn::Module {
    torch::Device ctx;
    std::shared_ptr<ResNet_prediction<BasicBlock>> network;
    Prediction(torch::Device& ctx) : ctx(ctx), network(resnet_prediction(ACTIONS)) {
        register_module("prediction", network);
    }

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
        return network->forward(x);
    }
};

struct Dynamics : torch::nn::Module {
    torch::Device ctx;
    std::shared_ptr<ResNet_dynamics<BasicBlock>> network;
    Dynamics(torch::Device& ctx) : ctx(ctx), network(resnet_dynamics(HIDDEN)) {
        register_module("dynamics", network);
    }

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x, torch::Tensor action) {
        return network->forward(x, action);
    }
};

struct Network : Network_i {
    torch::Device ctx;
    std::shared_ptr<Representation> representation;
    std::shared_ptr<Dynamics> dynamics;
    std::shared_ptr<Prediction> prediction;

    int _training_steps;
    bool is_train;

    Network(torch::Device ctx, bool train=false) : ctx(ctx), representation(std::make_shared<Representation>(ctx)),
        dynamics(std::make_shared<Dynamics>(ctx)), prediction(std::make_shared<Prediction>(ctx)), _training_steps(0), is_train(train) {
        to(ctx);
        _train(train);
    }

    std::vector<torch::Tensor> parameters() override {
        std::vector<torch::Tensor> params;
        std::vector<torch::Tensor> r_params = representation->parameters();
        std::vector<torch::Tensor> d_params = dynamics->parameters();
        std::vector<torch::Tensor> p_params = prediction->parameters();

        std::copy(r_params.begin(), r_params.end(), std::back_inserter(params));
        std::copy(d_params.begin(), d_params.end(), std::back_inserter(params));
        std::copy(p_params.begin(), p_params.end(), std::back_inserter(params));

        return params;
    }

    void _train(bool train) override {
        is_train = train;
        representation->train(train);
        dynamics->train(train);
        prediction->train(train);
    }

    int training_steps() override {
        return _training_steps;
    }

    int inc_trainint_steps() override {
        return _training_steps++;
    }

    void to(torch::Device& ctx) {
        representation->to(ctx);
        dynamics->to(ctx);
        prediction->to(ctx);
    }

    batch_out_t initial_inference(batch_in_t& batch) override {
        torch::Tensor hidden_state_tensor = representation->forward(batch.batch.to(ctx));
        DUMP_LOG(hidden_state_tensor.sizes());
        std::tuple<torch::Tensor, torch::Tensor> prediction_output = prediction->forward(hidden_state_tensor);

        torch::Tensor pt = hidden_state_tensor.to(get_cpu_ctx());
        DUMP_LOG(pt.sizes())

        torch::Tensor lt = std::get<0>(prediction_output).to(get_cpu_ctx());
        DUMP_LOG(lt.sizes())

        batch_out_t ret;
        ret.out = {};
        for(int id = 0; id < hidden_state_tensor.size(0); id++) {
            HiddenState_t hidden;

            if(!is_train) {
                auto acc = pt.accessor<float, 2>();
                for (int i = 0; i < HIDDEN; i++) {
                    auto a = acc[id][i];
                    assert(!isnan(a));
                    hidden.emplace_back(a);
                }
            }
            Policy_t policy;

            if(!is_train) {
                auto acc1 = lt.accessor<float, 3>();
                for (int i = 0; i < ACTIONS; i++) {
                    auto a = acc1[id][0][i];
                    assert(!isnan(a));
                    policy.emplace_back(a);
                }
            }
            float v = std::get<1>(prediction_output).to(get_cpu_ctx())[id].item<float>();
            assert(!isnan(v));
            std::shared_ptr<NetworkOutputTensor> tensor = nullptr;
            if(is_train) {
                tensor = NetworkOutputTensor::make_tensor(std::get<1>(prediction_output)[id], torch::tensor({(float) 0}).to(ctx),
                                                          std::get<0>(prediction_output)[id], hidden_state_tensor[id]);
            }
            ret.out.emplace_back(NetworkOutput(v, 0, policy, hidden, tensor));
        }
        return ret;
    }

    batch_out_t recurrent_inference(batch_in_t& batch) override {
        std::tuple<torch::Tensor, torch::Tensor> hidden_state_tensor = dynamics->forward(batch.batch.to(ctx),
                                                                                        torch::tensor(batch.actions).to(ctx));
        std::tuple<torch::Tensor, torch::Tensor>  prediction_output = prediction->forward(std::get<0>(hidden_state_tensor));

        torch::Tensor hidden_tensor = std::get<0>(hidden_state_tensor).to(get_cpu_ctx());
        DUMP_LOG(hidden_tensor.sizes())
        torch::Tensor lt = std::get<0>(prediction_output).to(get_cpu_ctx());
        DUMP_LOG(lt.sizes());
        batch_out_t ret;
        ret.out = {};
        for(int id = 0; id < hidden_tensor.size(0); id++) {
            HiddenState_t hidden;
            if(!is_train) {
                auto acc = hidden_tensor.accessor<float, 3>();
                for (int i = 0; i < HIDDEN; i++) {
                    auto a = acc[id][0][i];
                    assert(!isnan(a));
                    hidden.emplace_back(a);
                }
            }
            Policy_t policy;

            if(!is_train) {
                auto acc1 = lt.accessor<float, 3>();
                for (int i = 0; i < ACTIONS; i++) {
                    auto a = acc1[id][0][i];
                    assert(!isnan(a));
                    policy.emplace_back(a);
                }
            }

            float r = std::get<1>(hidden_state_tensor).to(get_cpu_ctx())[id].item<float>();
            float v = std::get<1>(prediction_output).to(get_cpu_ctx())[id].item<float>();
            assert(!isnan(r));
            assert(!isnan(v));
            std::shared_ptr<NetworkOutputTensor> tensor = nullptr;
            if(is_train) {
                tensor = NetworkOutputTensor::make_tensor(std::get<1>(prediction_output)[id], std::get<1>(hidden_state_tensor)[id],
                                                          std::get<0>(prediction_output)[id], std::get<0>(hidden_state_tensor)[id]);
            }
            ret.out.emplace_back( NetworkOutput(v, r, policy, hidden, tensor));
        }
        return ret;

    }

    void zero_grad() override {
        representation->zero_grad();
        dynamics->zero_grad();
        prediction->zero_grad();
    }


    void _save(std::string model_path, std::shared_ptr<torch::nn::Module> module) {
        std::ofstream of(model_path);
        boost::interprocess::file_lock ol(model_path.c_str());
        boost::interprocess::scoped_lock<boost::interprocess::file_lock> sol(ol);
        torch::save(module, model_path);

    }

    void save_network(int step, std::string filename) override {
        _save(filename + ".r.ckpt", representation);
        _save(filename + ".d.ckpt", dynamics);
        _save(filename + ".p.ckpt", prediction);
    }

    void _load(std::string model_path, std::shared_ptr<torch::nn::Module> module) {
        boost::interprocess::file_lock ol(model_path.c_str());
        boost::interprocess::scoped_lock<boost::interprocess::file_lock> sol(ol);
        torch::load(module, model_path);
    }

    void load_network(std::string filename) {
        _load(filename + ".r.ckpt", representation);
        _load(filename + ".d.ckpt", dynamics);
        _load(filename + ".p.ckpt", prediction);
    }
};


struct SingleInference : Inference_i {

    std::shared_ptr<batch_queue_in_t>& initial_queue_write;
    std::shared_ptr<batch_queue_in_t>& recurent_queue_write;

    std::shared_ptr<batch_queue_out_t> initial_queue_read;
    std::shared_ptr<batch_queue_out_t> recurent_queue_read;

    SingleInference(
            std::shared_ptr<batch_queue_in_t>& initial_queue_write,
            std::shared_ptr<batch_queue_in_t>& recurent_queue_write
            ) :
            initial_queue_write(initial_queue_write),
            recurent_queue_write(recurent_queue_write),
            initial_queue_read(std::make_shared<batch_queue_out_t>(10)),
            recurent_queue_read(std::make_shared<batch_queue_out_t>(10)) {
    }

    batch_out_t initial_inference(batch_in_t& batch) override {
        batch.out = {};
        batch.out.emplace_back(initial_queue_read);
        initial_queue_write->enqueue(batch);
        return initial_queue_read->dequeue();
    }

    batch_out_t recurrent_inference(batch_in_t& batch) override {
        batch.out = {};
        batch.out.emplace_back(recurent_queue_read);
        recurent_queue_write->enqueue(batch);
        return recurent_queue_read->dequeue();
    }
};


struct BatchInference : Inference_i {
    std::shared_ptr<batch_queue_in_t> initial_queue_read;
    std::shared_ptr<batch_queue_in_t> recurrent_queue_read;

    std::shared_ptr<Network_i> inference;
    std::vector<batch_in_t> initial_batches;
    std::vector<batch_in_t> recurrent_batches;
    int batch_size;

    std::atomic_bool run;

    std::shared_ptr<std::thread> i_thread, r_thread;
    std::thread::native_handle_type i_handle, r_handle;

    BatchInference(std::shared_ptr<Network_i> inference, int batch_size) : inference(inference),
                                                                              batch_size(batch_size),
                                                                              run(true),
                                                                              initial_queue_read(
                                                                                      std::make_shared<batch_queue_in_t>(
                                                                                              batch_size)),
                                                                              recurrent_queue_read(
                                                                                      std::make_shared<batch_queue_in_t>(
                                                                                              batch_size)),
                                                                                              initial_batches({}),
                                                                                              recurrent_batches({}){

        i_thread = std::make_shared<std::thread>([this](){initial_thread();});
        r_thread = std::make_shared<std::thread>([this](){recurrent_thread();});
        DUMP_LOG(batch_size)
    }

    virtual ~BatchInference() {
        run = false;
        i_thread->join();
        r_thread->join();
    }

    void initial_thread() {
        DUMP_LOG(batch_size)
#if 0
        while (run) {
                batch_in_t b = initial_queue_read->dequeue();
                initial_batches.emplace_back(b);

                DUMP_LOG(initial_batches.size())

                if(initial_batches.size() == batch_size) {
                    batch_in_t batch_in = merge_batch(initial_batches);
                    batch_out_t batch_out = initial_inference(batch_in);
                    distill_batch(batch_in,batch_out);
                    initial_batches.clear();
                }
        }
#else
        while (run) {
            std::vector<batch_in_t> b = initial_queue_read->dequeue_all(std::chrono::milliseconds(100));
            if(b.empty())
                continue;
            std::copy(b.begin(), b.end(), std::back_inserter(initial_batches));

            batch_in_t batch_in = merge_batch(initial_batches);
            batch_out_t batch_out = initial_inference(batch_in);
            distill_batch(batch_in,batch_out);
            initial_batches.clear();
        }
#endif
    }

    void recurrent_thread() {
        DUMP_LOG(batch_size)
#if 0
        while (run) {
            batch_in_t b = recurrent_queue_read->dequeue();
            recurrent_batches.emplace_back(b);
            DUMP_LOG(recurrent_batches.size())


            if (recurrent_batches.size() == batch_size) {
                batch_in_t batch_in = merge_batch(recurrent_batches);
                batch_out_t batch_out = recurrent_inference(batch_in);
                distill_batch(batch_in, batch_out);
                recurrent_batches.clear();
            }
        }
#else
        while (run) {
            std::vector<batch_in_t> b = recurrent_queue_read->dequeue_all(std::chrono::milliseconds(100));
            if(b.empty())
                continue;
            std::copy(b.begin(), b.end(), std::back_inserter(recurrent_batches));
            DUMP_LOG(recurrent_batches.size())

            batch_in_t batch_in = merge_batch(recurrent_batches);
            batch_out_t batch_out = recurrent_inference(batch_in);
            distill_batch(batch_in, batch_out);
            recurrent_batches.clear();
        }
#endif
    }

    batch_in_t merge_batch(std::vector<batch_in_t>& batch) {
        batch_in_t ret;
        ret.out = {};
        ret.actions = {};
        ret.out.emplace_back(batch[0].out[0]);
        if(batch[0].actions.size() > 0) {
            assert(batch[0].actions.size() == 1);
            ret.actions.emplace_back(batch[0].actions[0]);
        }
        torch::Tensor t = batch[0].batch;
        for(int i = 1; i < batch.size(); i++) {
            t = torch::cat({t, batch[i].batch});
            ret.out.emplace_back(batch[i].out[0]);
            if(batch[i].actions.size() > 0) {
                assert(batch[0].actions.size() == 1);
                ret.actions.emplace_back(batch[i].actions[0]);
            }

        }
        ret.batch = t;
        return ret;
    }

    void distill_batch(batch_in_t& batch_in, batch_out_t& batch_out) {
        DUMP_LOG(batch_out.out.size());
//#pragma omp parallel for
        if(batch_in.out.size() == batch_out.out.size()) {
            for(int i = 0; i < batch_in.out.size(); i++) {
                    batch_out_t b;
                    b.out.emplace_back(batch_out.out[i]);
                    batch_in.out[i]->enqueue(b);
            }
        } else {
            batch_in.out[0]->enqueue(batch_out);
        }
    }


    batch_out_t initial_inference(batch_in_t& batch) override {
        return inference->initial_inference(batch);
    }

    batch_out_t recurrent_inference(batch_in_t& batch) override {
        return inference->recurrent_inference(batch);
    }

};


struct SingleInferenceWrapper : Network_i {
    std::shared_ptr<SingleInference> single_inference;
    std::shared_ptr<Network_i> network;

    SingleInferenceWrapper(std::shared_ptr<SingleInference> single_inference, std::shared_ptr<Network_i> network) :
        single_inference(single_inference),
        network(network) {

    }

    batch_out_t initial_inference(batch_in_t& batch) override {
        return single_inference->initial_inference(batch);
    }

    batch_out_t recurrent_inference(batch_in_t& batch) override {
        return single_inference->recurrent_inference(batch);
    }

    void save_network(int step, std::string path) override {
        network->save_network(step, path);
    }

    void load_network(std::string path) override {
        network->load_network(path);
    }

    int training_steps() override {
        return network->training_steps();
    }

    void _train(bool t) override {
        network->_train(t);
    }

    std::vector<torch::Tensor> parameters() override {
        return network->parameters();
    }

    void zero_grad() override {
        network->zero_grad();
    }

    int inc_trainint_steps() override {
        return network->inc_trainint_steps();
    }
};


struct BatchSharedStorage : SharedStorage_i {

    const std::string path;

    BatchSharedStorage(const MuZeroConfig &config) : path(config.path) {

    }

    std::shared_ptr<Network_i> latest_network(torch::Device ctx) override {
        if (boost::filesystem::is_empty(path + "/network")) {
            return make_uniform_network(ctx);
        }
        std::shared_ptr<Network_i> network = make_uniform_network(ctx);
        network->load_network(path + "/network/latest");
        return network;
    }

    void save_network(int step, std::shared_ptr<Network_i>& network) override {
        network->save_network(step, path + "/network/latest");
    }

    std::shared_ptr<Network_i> make_uniform_network(torch::Device& ctx) {
        return std::make_shared<Network>(ctx);
    }
};

struct SingleSharedStorage : SharedStorage_i {
    std::shared_ptr<SharedStorage_i> shared_storage;
    std::vector<std::shared_ptr<BatchInference>> batch_inference;
    int batch_size;
    int loop_counter;
    int batch_inference_count;

    SingleSharedStorage(std::shared_ptr<SharedStorage_i> shared_storage, int batch_size, torch::Device ctx, int batch_inference_count) :
        shared_storage(shared_storage),
                                                           batch_inference({}), batch_inference_count(batch_inference_count),
                                                           batch_size(batch_size), loop_counter(0) {

        if(batch_inference.size() == 0) {
            for(int i = 0; i < batch_inference_count; i++) {
                std::shared_ptr<Network_i> network = shared_storage->latest_network(ctx);
                std::shared_ptr<BatchInference> batch_inference_i = std::make_shared<BatchInference>(network, batch_size);
                batch_inference.emplace_back(batch_inference_i);
            }
        }

    }

    std::shared_ptr<Network_i> latest_network(torch::Device ctx) override {
        std::shared_ptr<SingleInference> single_inference = std::make_shared<SingleInference>(batch_inference[loop_counter]->initial_queue_read,
                batch_inference[loop_counter]->recurrent_queue_read);
        std::shared_ptr<Network_i> ret = std::make_shared<SingleInferenceWrapper>(single_inference, batch_inference[loop_counter]->inference);
        loop_counter++;
        loop_counter %= batch_inference_count;
        return ret;
    }

    void save_network(int step, std::shared_ptr<Network_i>& network) override {
        shared_storage->save_network(step, network);
    }
};



int softmax_sample(std::vector<float> visit_counts, float temperature) {

    if( temperature == 0) {
        return std::max_element(visit_counts.begin(), visit_counts.end()) - visit_counts.begin();
    }
    float counts_sum = std::accumulate(visit_counts.begin(), visit_counts.end(), (float) 0.,
                                       [temperature](float &a, float &b) { return a + std::pow(b, 1./temperature); });
    std::vector<float> d;
    for (int i = 0; i < visit_counts.size(); i++) {
        float s = std::pow(visit_counts[i], 1./temperature) / counts_sum;
        d.emplace_back(s);
    }



    std::discrete_distribution<int> distribution(d.begin(), d.end());

    return distribution(generator);

}

void add_exploration_noise(MuZeroConfig& config, std::shared_ptr<Node>& node) {
    auto n = rnd.Dirichlet<ACTIONS>(config.root_dirichlet_alpha);
    float frac = config.root_exploration_fraction;
    for(int i = 0;i < node->children.size(); i++) {
        node->children[i]->prior = node->children[i]->prior * (1-frac) + n[i] * frac;
        assert(!isnan(node->children[i]->prior));
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

void expand_node(std::shared_ptr<Node>& node, Player to_play, ActionList_t actions, NetworkOutput& network_output) {
    node->to_play = to_play.id;
    node->hidden_state = network_output.hidden_state;
    node->reward = network_output.reward;
    float policy_sum = std::accumulate(network_output.policy_logits.begin(), network_output.policy_logits.end(), (float)0,
            [](float &a, float &b ){ return a + std::exp(b) ;});
    for(int i = 0;i < network_output.policy_logits.size(); i++) {
        float e = std::exp(network_output.policy_logits[i]);
        node->children.emplace_back(std::make_shared<Node>(e/policy_sum));
    }
}


float ucb_score(MuZeroConfig& config, std::shared_ptr<Node>& parent, std::shared_ptr<Node>& child, MinMaxStats& min_max_stats) {
    float pb_c = std::log(((float)parent->visit_count + config.pb_c_base + 1) /
            config.pb_c_base) + config.pb_c_init;
    pb_c *= std::sqrt((float)parent->visit_count) / ((float)child->visit_count + 1);

    float prior_score = pb_c * child->prior;
    float value_score = min_max_stats.normalize(child->value());
    assert(!isnan(value_score));
    assert(!isnan(prior_score));
    return prior_score + value_score;
}

std::pair<Action, std::shared_ptr<Node>> select_child(MuZeroConfig& config, std::shared_ptr<Node>& node,
        MinMaxStats& min_max_stats) {
    float _ucb_score = -MAXIMUM_FLOAT_VALUE;
    int action = -1;
    std::shared_ptr<Node> child;
    for(int i = 0; i < node->children.size(); i++) {
        float u = ucb_score(config, node, node->children[i], min_max_stats);
        assert(!isnan(u));
        if (_ucb_score < u) {
            _ucb_score = u;
            action = i;
            child = node->children[i];
        }
    }

    assert(child != nullptr);
    return {Action(action), child};
}

Action select_action(MuZeroConfig& config, int num_moves, std::shared_ptr<Node>& node, std::shared_ptr<Network_i>& network) {
    std::vector<float> visit_counts;
    for(int i = 0;i < node->children.size(); i++) {
        visit_counts.emplace_back(node->children[i]->visit_count);
    }
    float t = config.visit_softmax_temperature_fn->operator()(num_moves, network->training_steps());
    int action = softmax_sample(visit_counts, t);
    return Action(action);

}

void run_mcts(MuZeroConfig& config, std::shared_ptr<Node>& root, ActionHistory action_history, std::shared_ptr<Network_i>& network) {
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

        assert(search_path.size() - 2 >= 0);
        std::shared_ptr<Node> parent = search_path[search_path.size() - 2];
        batch_in_s batch = batch_in_s::make_batch(parent->hidden_state, history.last_action());
        batch_out_t batch_out = network->recurrent_inference(batch);
        NetworkOutput network_output = batch_out.network_output();
        expand_node(node, history.to_play(), history.action_space(), network_output);

        backpropagate(search_path, network_output.value, history.to_play(), config.discount, min_max_stats);

    }
}

std::shared_ptr<Game> play_game(MuZeroConfig& config, std::shared_ptr<Network_i> network) {
    std::shared_ptr<Game> game = config.new_game();

    int count = 0;
    while(!game->terminal() && game->history.size() < config.max_moves) {
        std::shared_ptr<Node> root(std::make_shared<Node>(0));
        Image_t current_observation = game->make_image(-1);
        batch_in_s batch = batch_in_s::make_batch(current_observation);
        NetworkOutput no = network->initial_inference(batch).network_output();
        expand_node(root, game->to_play(), game->legal_actions(), no);
        add_exploration_noise(config, root);

//        std::cout << "run_mcts " << count << std::endl;
        count++;
        run_mcts(config, root, game->action_history(), network);
        Action action = select_action(config, game->history.size(), root, network);
        game->apply(action);
        game->store_search_statistics(root);
    }

    return game;
}

void run_selfplay(MuZeroConfig config, std::shared_ptr<SharedStorage_i> storage, ReplayBuffer replay_buffer, int tid) {

    for(int i = 0;i < config.num_selfplay; i++) {
        std::shared_ptr<Network_i> network = storage->latest_network(get_ctx());
        std::shared_ptr<Game> game = play_game(config, network);
        replay_buffer.save_game(game);
    }
}

torch::Tensor cross_entropy_loss(torch::Tensor input, torch::Tensor target, int step) {
    torch::Tensor t = input - input.max_values(1).view({-1, 1});
    torch::Tensor r = -(target * (t - torch::log(torch::exp(t).sum(1)).view({-1,1}))).sum(1);
    torch::Tensor entropy = -(target * torch::log(target+1e-12)).sum(1).mean();
    if(step % 10 == 0)
        std::cout << step << " entropy: " << entropy.item<float>() << " cross_entropy: " << r.mean().item<float>();
    return r;
//    return -(target *(torch::log_softmax(input, 1))).sum(1).mean();
}

struct CustomCat {

    torch::Tensor tensor;
    bool set;

    CustomCat(torch::Tensor tensor) : tensor(tensor), set(true) {

    }

    CustomCat() : set(false){

    }

    CustomCat& operator+=(const CustomCat& other) {
        if(!set) {
            tensor = other.tensor;
            set = true;
        } else {
            tensor = torch::cat({tensor, other.tensor});
        }
        return *this;
    }
};




void update_weights(torch::optim::Optimizer& opt, std::shared_ptr<Network_i>& network, std::vector<Batch> batch, float weight_decay, torch::Device ctx, int step) {

    torch::Tensor values_v;
    std::vector<float> target_values_v;
    torch::Tensor rewards_v;
    std::vector<float> target_rewards_v;
    torch::Tensor logits_v;
    torch::Tensor target_policies_v;
    torch::Tensor scale_v;



    std::vector<NetworkOutput> network_output_vector(batch.size());
    std::vector<std::vector<NetworkOutput>> predictions(batch.size());
    std::vector<std::vector<float>> scales_b(batch.size());

    DUMP_LOG(batch.size())
//#pragma omp parallel num_threads(12)  default(none) shared(batch, network, network_output_vector, predictions, scales_b)
//    {
//#pragma omp for schedule(static)
        std::vector<Image_t> img;
        for (int i = 0; i < batch.size(); ++i) {
            Image_t image = batch[i].image;
            img.emplace_back(image);
        }

            batch_in_s b = batch_in_s::make_batch(img);
            batch_out_t network_output = network->initial_inference(b);

            for(int i = 0; i < batch.size(); i++) {
//                network_output_vector[i] = network_output[i];

                predictions[i].emplace_back(network_output.out[i]);
                scales_b[i].emplace_back(1);
            }
//        }
//    }

#pragma omp parallel num_threads(64) default(none) shared(batch, network, network_output_vector, predictions, scales_b, network_output)
    {
#pragma omp for schedule(dynamic)
        for (int i = 0; i < batch.size(); i++) {

            ActionList_t actions = batch[i].action;

            torch::Tensor hidden_state = network_output.out[i].tensor->hidden_tensor;
            for (int j = 0; j < actions.size(); j++) {

                batch_in_s b = batch_in_s::make_batch(hidden_state, actions[j], true);
                NetworkOutput network_output_1 = network->recurrent_inference(b).network_output();
                predictions[i].emplace_back(network_output_1);
                scales_b[i].emplace_back(1. / actions.size());
//                scale_v = torch::cat({scale_v, torch::tensor((float) 1./actions.size())});
                hidden_state = network_output_1.tensor->hidden_tensor;
            }
        }
    }
#if 1
//    CustomCat values_cat_all(predictions[0][0].tensor->value_tensor.reshape({1}));
//    CustomCat rewards_cat_all(predictions[0][0].tensor->reward_tensor.reshape({1}));
//    CustomCat logits_cat_all(predictions[0][0].tensor->policy_tensor.reshape({1, ACTIONS}));
//    CustomCat target_policies_cat_all(torch::tensor(batch[0].target[0].policy).reshape({1, ACTIONS}));

    CustomCat values_cat_all;
    CustomCat rewards_cat_all;
    CustomCat logits_cat_all;
    CustomCat target_policies_cat_all;
    CustomCat scale_cat_all;

#pragma omp declare reduction(custom: CustomCat : omp_out += omp_in)
#pragma omp parallel for num_threads(64) reduction(custom:values_cat_all) reduction(custom:rewards_cat_all) reduction(custom:logits_cat_all) reduction(custom:target_policies_cat_all) reduction(custom:scale_cat_all)
    for (int i = 0; i < batch.size(); i++) {
        int start_index = 0;
//        if(i == 0) {
//            start_index = 1;
//        }
        if(start_index < predictions[i].size()) {
            std::vector<Target> targets = batch[i].target;
//            CustomCat values_cat(predictions[i][start_index].tensor->value_tensor.reshape({1}));
//            CustomCat rewards_cat(predictions[i][start_index].tensor->reward_tensor.reshape({1}));
//            CustomCat logits_cat(predictions[i][start_index].tensor->policy_tensor.reshape({1, ACTIONS}));
//            CustomCat target_policies_cat(torch::tensor(targets[start_index].policy).reshape({1, ACTIONS}));

            CustomCat values_cat;
            CustomCat rewards_cat;
            CustomCat logits_cat;
            CustomCat target_policies_cat;
            CustomCat scale_cat;


//#pragma omp parallel for reduction(custom:values_cat) reduction(custom:rewards_cat) reduction(custom:logits_cat) reduction(custom:target_policies_cat)
            for (int k = start_index; k < std::min(predictions[i].size(), targets.size()); k++) {
                values_cat += predictions[i][k].tensor->value_tensor.view({1});
                rewards_cat += predictions[i][k].tensor->reward_tensor.view({1});
                logits_cat += predictions[i][k].tensor->policy_tensor.view({1, ACTIONS});
                target_policies_cat += torch::tensor(targets[k].policy).view({1, ACTIONS});
                scale_cat += torch::tensor((float) scales_b[i][k]);
            }


            values_cat_all += values_cat;
            rewards_cat_all += rewards_cat;
            logits_cat_all += logits_cat;
            target_policies_cat_all += target_policies_cat;
            scale_cat_all += scale_cat;
        }
    }


    for (int i = 0; i < batch.size(); i++) {

        Image_t image = batch[i].image;

        std::vector<Target> targets = batch[i].target;


        for (int k = 0; k < std::min(predictions[i].size()  , targets.size()); k++) {
//            if (i == 0 && k == 0) {
//                scale_v = torch::tensor((float) scales_b[i][k]);
//            } else {
//                scale_v = torch::cat({scale_v, torch::tensor((float) scales_b[i][k])});
//            }

            target_values_v.emplace_back(targets[k].value);
            target_rewards_v.emplace_back(targets[k].reward);

        }
    }


        torch::Tensor values = values_cat_all.tensor.view({-1, 1}).to(ctx);
        torch::Tensor target_values = torch::tensor(target_values_v).view({-1, 1}).to(ctx);
        torch::Tensor rewards = rewards_cat_all.tensor.view({-1, 1}).to(ctx);
        torch::Tensor target_rewards = torch::tensor(target_rewards_v).view({-1, 1}).to(ctx);
        torch::Tensor logits = logits_cat_all.tensor.to(ctx);
        torch::Tensor target_policies = target_policies_cat_all.tensor.to(ctx);
        torch::Tensor scale = scale_cat_all.tensor.view({-1, 1}).to(ctx);

//        std::cout << values.sizes() << std::endl;
//        std::cout << target_values.sizes() << std::endl;
        torch::Tensor l = (((values - target_values).pow(2).mean(1)
                            + (rewards - target_rewards).pow(2).mean(1)
                            + cross_entropy_loss(logits, target_policies, step)) * scale).mean();
//        std::cout << target_policies.sizes() << std::endl;
//        torch::Tensor l = (cross_entropy_loss(logits, target_policies)).mean();

        if(step % 10 == 0)
            std::cout << "\t\t loss: " << l.item<float>() << std::endl;

        opt.zero_grad();
        l.backward();
        opt.step();


#endif

}

void train_network(MuZeroConfig& config, std::shared_ptr<SharedStorage_i> storage, ReplayBuffer replay_buffer, torch::Device ctx) {
    std::shared_ptr<Network_i> network = storage->latest_network(ctx);
    network->_train(true);

    std::vector<torch::Tensor> params = network->parameters();

    torch::optim::Adam opt(params, torch::optim::AdamOptions(config.lr_init)
    /*.momentum(config.momentum)*/.weight_decay(config.weight_decay));

    for(int i = 0; i < config.training_steps; i++) {
//        std::cout << i << "\t";
        std::cout << "*" << "\t";
        if(i != 0 && i % config.checkpoint_interval == 0) {
            storage->save_network(i, network);
        }
        std::vector<Batch> batch = replay_buffer.sample_batch(config.num_unroll_steps, config.td_steps);
        update_weights(opt, network, batch, config.weight_decay, ctx, i);
        network->inc_trainint_steps();
    }
    storage->save_network(config.training_steps, network);

}


std::shared_ptr<Network_i> muzero(MuZeroConfig config) {
    std::shared_ptr<BatchSharedStorage> b_storage = std::make_shared<BatchSharedStorage>(config);
    int bs = config.train ? 128 : config.batch_size;//config.num_actors;
    std::shared_ptr<SingleSharedStorage> storage = std::make_shared<SingleSharedStorage>(b_storage, bs, get_ctx(), config.train ? 1 : config.num_executors);
    ReplayBuffer replay_buffer(config);

    std::vector<std::shared_ptr<std::thread>> threads;
    for(int i = 0; i < config.num_actors; i++) {
//        run_selfplay(config, storage, replay_buffer, i);
        threads.emplace_back(std::make_shared<std::thread>(run_selfplay, config, storage, replay_buffer, i));
    }


    if(config.train)
        train_network(config, storage, replay_buffer, get_train_ctx());

    for(int i = 0; i < threads.size(); i++)
        threads[i]->join();

    return storage->latest_network(get_train_ctx());
}




int main(int argc, char** argv) {
    boost::program_options::options_description desc("Allowed options");
    desc.add_options()
    ("help", "produce help")
            ("train", boost::program_options::value<bool>(), "set train mode")
            ("workers", boost::program_options::value<int>(), "number of workers")
            ("lr", boost::program_options::value<float>(), "learning rate")
            ("path", boost::program_options::value<std::string>(), "data path")
            ("window", boost::program_options::value<int>(), "window size")
            ("training_steps", boost::program_options::value<int>(), "number of training steps")
            ("checkpoint_interval", boost::program_options::value<int>(), "number of steps when to checkpoint network")
            ("num_selfplay", boost::program_options::value<int>(), "number of games to play")
            ("batch", boost::program_options::value<int>(), "batch size")
            ("executors", boost::program_options::value<int>(), "number of executors");

    boost::program_options::variables_map vm;
    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
    boost::program_options::notify(vm);
    MuZeroConfig config = make_c_config();

    if(vm.count("workers")) {
        config.num_actors = vm["workers"].as<int>();
    } else {
        config.num_actors = 0;
    }

    if(vm.count("train")) {
        config.train = vm["train"].as<bool>();
    } else {
        config.train = false;
    }

    if(vm.count("lr")) {
        config.lr_init = vm["lr"].as<float>();
    }

    if(vm.count("path")) {
        config.path = vm["path"].as<std::string>();
    } else {
        config.path = "/home/tomas/CLionProjects/muzero";
    }

    if(vm.count("window")) {
        config.window_size = vm["window"].as<int>();
    } else {
        config.window_size = 40000;
    }

    if(vm.count("training_steps")) {
        config.training_steps = vm["training_steps"].as<int>();
    }

    if(vm.count("checkpoint_interval")) {
        config.checkpoint_interval = vm["checkpoint_interval"].as<int>();
    }
    if(vm.count("num_selfplay")) {
        config.num_selfplay = vm["num_selfplay"].as<int>();
    }
    if(vm.count("batch")) {
        config.batch_size = vm["batch"].as<int>();
    }

    if(vm.count("executors")) {
        config.num_executors = vm["executors"].as<int>();
    }
    omp_set_dynamic(0);
    omp_set_num_threads(64);
    DUMP_LOG(omp_get_max_threads());
    muzero(config);
    return 0;
}