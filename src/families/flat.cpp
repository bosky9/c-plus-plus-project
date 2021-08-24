#include "../../include/families/flat.hpp"

Flat::Flat(const std::string& transform) : Family{transform}, _covariance_prior{false} {}

Flat::Flat(const Flat& flat) : Family(flat) {
    _covariance_prior = flat._covariance_prior;
}

Flat::Flat(Flat&& flat) : Family(std::move(flat)) {
    _covariance_prior      = flat._covariance_prior;
    flat._covariance_prior = false;
}

Flat& Flat::operator=(const Flat& flat) {
    if (this == &flat)
        return *this;
    Family::operator  =(flat);
    _covariance_prior = flat._covariance_prior;
    return *this;
}

Flat& Flat::operator=(Flat&& flat) {
    _covariance_prior      = flat._covariance_prior;
    flat._covariance_prior = false;
    Family::operator       =(std::move(flat));
    return *this;
}

bool operator==(const Flat& flat1, const Flat& flat2) {
    return is_equal(flat1, flat2) && flat1._covariance_prior == flat2._covariance_prior;
}

double Flat::logpdf(double mu) {
    return 0.0;
}