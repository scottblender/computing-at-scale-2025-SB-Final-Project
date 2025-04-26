#ifndef PROPAGATION_SETTINGS_HPP
#define PROPAGATION_SETTINGS_HPP

struct PropagationSettings {
    double mu;
    double F;
    double c;
    double m0;
    double g0;
    int num_eval_per_step;
    int num_subintervals;
    int state_size;
    int control_size;
};

#endif // PROPAGATION_SETTINGS_HPP
