#include "latent_variables.hpp"

#include "output/tableprinter.hpp"
#include <string>
#include <vector>
#include <utility>
#include <list>
#include <tuple>
#include <numeric>

LatentVariable::LatentVariable(const std::string& name, const Family& prior, const Family& q) :
      _name{name},
      _prior{prior},
      _transform{prior.get_transform()},
      _q{q}
{}

void LatentVariable::plot_z(double width, double height) {
    assert(_sample || (_value && _std));
    if (_sample) {
        //TODO
    }
    else {
        //TODO
    }
}

std::string LatentVariable::get_name() const {
    return _name;
}

Family LatentVariable::get_prior() const {
    return _prior;
}

double LatentVariable::get_start() const {
    return _start;
}

std::optional<double> LatentVariable::get_value() const {
    return _value;
}

Family LatentVariable::get_q() const {
    return _q;
}

void LatentVariable::set_prior(const Family& prior) {
    _prior = prior;
}

void LatentVariable::set_start(double start) {
    _start = start;
}

LatentVariables::LatentVariables(const std::string& model_name) :
      _model_name{model_name}
{
    _z_list = {};
    _z_indices = {};
}

inline std::ostream& operator<<(std::ostream& stream, const LatentVariables& lvs) {
    std::vector<std::string> z_names = lvs.get_z_names();
    std::vector<Family> priors = lvs.get_z_priors();
    std::pair<std::vector<std::string>,std::vector<std::string>> z_priors_names = lvs.get_z_priors_names();
    std::vector<std::string> prior_names = z_priors_names.first;
    std::vector<std::string> prior_z_names = z_priors_names.second;
    std::vector<std::string> vardist_names = lvs.get_z_approx_dist_names();
    std::vector<std::string> transforms = lvs.get_z_transforms_names();

    std::list<std::tuple<std::string, std::string, int>> fmt = {
            std::make_tuple("Index","z_index",8),
            std::make_tuple("Latent Variable","z_name",25),
            std::make_tuple("Prior","z_prior",15),
            std::make_tuple("Prior Hyperparameters","z_hyper",25),
            std::make_tuple("V.I. Dist","z_vardist",10),
            std::make_tuple("Transform","z_transform",10)
    };

    std::list<std::map<std::string, std::string>> z_row;
    for (size_t z{0}; z < lvs._z_list.size(); z++)
        z_row.push_back({
                {"z_index", std::to_string(z)},
                {"z_name", z_names.at(z)},
                {"z_prior", prior_names.at(z)},
                {"z_hyper", prior_z_names.at(z)},
                {"z_vardist", vardist_names.at(z)},
                {"z_transform", transforms.at(z)}
        });

    stream << TablePrinter(fmt, " ","=")._call_(z_row);
    return stream;
}

void LatentVariables::add_z(const std::string& name, const Family& prior, const Family& q, bool index) {
    LatentVariable lv{name,prior,q};
    _z_list.push_back(std::move(lv));
    if (index)
        _z_indices[name] = { {"start",_z_list.size()-1}, {"end",_z_list.size()-1} };
}

void LatentVariables::create(const std::string& name, const std::vector<size_t>& dim, const Family& prior, const Family& q) {
    // Initialize indices vector
    size_t indices_dim = std::accumulate(dim.begin(),dim.end(),1,std::multiplies<>());
    std::vector<std::string> indices(indices_dim,"(");
    for (size_t d{0}; d < dim.size(); d++) {
        size_t span = std::accumulate(dim.begin() + d + 1, dim.end(), 1, std::multiplies<>());
        std::string separator = (d == dim.size()-1) ? "," : ")";
        for (size_t index{0}; index < indices_dim; index++)
            indices.at(index) += std::to_string(index / span) + separator;
    }

    size_t starting_index = _z_list.size();
    _z_indices[name] = { {"start",starting_index}, {"end",starting_index+indices.size()-1}, {"dim",dim.size()} };
    for (const std::string& index : indices)
        add_z(name+" "+index, prior, q, false);
}

void LatentVariables::adjust_prior(const std::vector<size_t>& index, const Family& prior) {
    for (size_t item : index) {
        assert(item < 0 || item > _z_list.size()-1);
        _z_list.at(item).set_prior(prior);
        if (auto mu0 = _z_list.at(item).get_prior().get_mu0())
            _z_list.at(item).set_start(mu0.value());
        if (auto loc0 = _z_list.at(item).get_prior().get_loc0())
            _z_list.at(item).set_start(loc0.value());
    }
}

std::vector<std::string> LatentVariables::get_z_names() const {
    std::vector<std::string> names;
    for (const LatentVariable& z : _z_list)
        names.push_back(z.get_name());
    return names;
}

std::vector<Family> LatentVariables::get_z_priors() const {
    std::vector<Family> priors;
    for (const LatentVariable& z : _z_list)
        priors.push_back(z.get_prior());
    return priors;
}

std::pair<std::vector<std::string>,std::vector<std::string>> LatentVariables::get_z_priors_names() const {
    std::vector<Family> priors = get_z_priors();
    std::vector<std::string> prior_names;
    std::vector<std::string> prior_z_names;
    for (const Family& prior : priors) {
        prior_names.push_back(prior.get_name());
        prior_z_names.push_back(prior.get_z_name());
    }
    return {prior_names, prior_z_names};
}

std::vector<std::function<double(double)>> LatentVariables::get_z_transforms() const {
    std::vector<std::function<double(double)>> transforms;
    for (const LatentVariable& z : _z_list)
        transforms.push_back(z.get_prior().get_transform());
    return transforms;
}

std::vector<std::string> LatentVariables::get_z_transforms_names() const {
    std::vector<std::string> transforms;
    for (const LatentVariable& z : _z_list)
        transforms.push_back(z.get_prior().get_transform_name());
    return transforms;
}

Eigen::VectorXd LatentVariables::get_z_starting_values(bool transformed) const {
    std::vector<std::function<double(double)>> transforms = get_z_transforms();
    Eigen::VectorXd values(_z_list.size());
    for (size_t i{0}; i < _z_list.size(); i++) {
        values(i) = transformed ? transforms.at(i)(_z_list.at(i).get_start()) : _z_list.at(i).get_start();
    }
    return values;
}

Eigen::VectorXd LatentVariables::get_z_values(bool transformed) const {
    assert(_estimated);
    std::vector<std::function<double(double)>> transforms = get_z_transforms();
    Eigen::VectorXd values(_z_list.size());
    for (size_t i{0}; i < _z_list.size(); i++) {
        assert(_z_list.at(i).get_value());
        values(i) = transformed ? transforms.at(i)(_z_list.at(i).get_value().value()) : _z_list.at(i).get_value().value();
    }
    return values;
}

std::vector<Family> LatentVariables::get_z_approx_dist() const {
    std::vector<Family> dists;
    for (const LatentVariable& z : _z_list)
        dists.push_back(z.get_q());
    return dists;
}

std::vector<std::string> LatentVariables::get_z_approx_dist_names() const {
    std::vector<Family> approx_dists = get_z_approx_dist();
    std::vector<std::string> q_list(approx_dists.size());
    for (const Family& approx : approx_dists)
        q_list.emplace_back((approx.get_name() == "Normal") ? "Normal" : "Approximate distribution not detected");
    return q_list;
}