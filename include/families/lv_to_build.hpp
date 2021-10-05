#pragma once

#include "families/flat.hpp"
#include "families/normal.hpp"

struct Lv_to_build final {
    std::string _name;
    std::unique_ptr<Flat> _flat;
    std::unique_ptr<Normal> _normal; /**< Using a unique pointer avoids double deletes;
                                      *   the pointer is needed because otherwise parameter n could not be initialized,
                                      *   since it is of type Normal and it is inside the Normal class.
                                      */
    double _value;

    Lv_to_build(const std::string& name, Flat* flat, Normal* normal, double value);
    Lv_to_build(const Lv_to_build& build);
    Lv_to_build(Lv_to_build&& build);
    Lv_to_build& operator=(const Lv_to_build& build);
    Lv_to_build& operator=(Lv_to_build&& build);
    ~Lv_to_build();
}; /**<  Necessary for "build_latent_variables()" function.
    *   The python code appends to a list another list, this one:
    *   (['Normal Scale', Flat(transform='exp'), Normal(0, 3), 0.0])
    *   To translate the list above, we used this structure.
    */