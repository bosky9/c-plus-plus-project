#pragma once

#include "headers.hpp"

void metropolis_sampler(int sims_to_do, Eigen::MatrixXd& phi, const std::function<double(Eigen::VectorXd)>& posterior,
                        Eigen::VectorXd& a_rate, const Eigen::MatrixXd& rnums, const Eigen::VectorXd& crit);