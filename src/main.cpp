#include "families/family.hpp"
#include "families/normal.hpp"
#include "headers.hpp"

using namespace std;

int main() {
    // Tests for Family
    Family f{"exp"};

    cout << f._transform(5) << endl;
    cout << f._itransform(f._transform(5)) << endl;

    // Tests for Flat
    Flat flat{"exp"};
    cout << flat.logpdf(0) << endl;

    // Tests for Normal
    Normal n{2, 3};

    vector<double> ve{};
    vector<double> v{1, 2, 3};
    Eigen::MatrixXd M{};
    Eigen::MatrixXd* H_mu = n.approximating_model(ve, M, M, M, M, 2, v);
    cout << "approximating_model: " << H_mu << endl;

    H_mu = n.approximating_model_reg(ve, M, M, M, M, 2, v, ve, 0);
    cout << "approximating_model_reg: " << H_mu << endl;

    list<Normal::lv_to_build> lv = n.build_latent_variables();
    cout << "build_latent_variables: " << lv.front().name << " " << lv.front().n->vi_return_param(0) << " "
         << lv.front().n->vi_return_param(1) << " " << lv.front().zero << endl;

    vector<double> var = n.draw_variable(1, 2, 0, 0, 2);
    cout << "draw_variable: " << var[0] << " " << var[1] << endl;

    var = n.draw_variable_local(2);
    cout << "draw_variable_local: " << var[0] << " " << var[1] << endl;

    cout << "logpdf: " << n.logpdf(5) << endl;

    vector<double> mean{4, 1, 4};
    var = Normal::markov_blanket(v, mean, 1, 0, 0);
    cout << "markov_blanket: " << var[0] << " " << var[1] << " " << var[2] << endl;

    FamilyAttributes fa = Normal::setup();
    cout << "setup: " << fa.name << " " << fa.scale << endl;//...

    cout << "neg_loglikelihood: " << Normal::neg_loglikelihood(v, mean, 1, 0, 0) << endl;

    cout << "pdf: " << n.pdf(1) << endl;

    n.vi_change_param(0, 2);
    cout << "vi_change_param + vi_return_param: " << n.vi_return_param(0) << endl;
    cout << "vi_score: " << n.vi_score(2, 0) << " " << n.vi_score(2, 1) << endl;

    Normal n2{n};
    Normal n3;
    n3 = n;
    Normal n4{std::move(n)};
    Normal n5;
    n5 = std::move(n2);
}