var addon = require("../native");

class ExoticOptions {
    constructor(s0, exercise_price, years, risk_free_rate, volatility, tao) {
        this.s0 = s0;
        this.X = exercise_price;
        this.T = years;
        this.r = risk_free_rate;
        this.sigma = volatility;
        this.tao = tao
        this.price = 0;
    }

    americanCall(num_steps) {
       this.price = addon.binomial_american_call(this.s0, this.X, this.T, this.r, this.sigma, num_steps);
       console.log(this.price)
       return this;
    } 
    bermudanCall(num_steps, early_dates) {
        this.price = addon.bermudan_call(this.s0, this.X, this.T, this.r, this.sigma, num_steps, early_dates);
        console.log(this.price)
        return this;
     } 
    chooserOption(choice="call", tao) {
        if (choice == "call") {
            this.price = addon.european_call(this.s0, this.X, this.T, this.r, this.sigma, this.T);
        } else if (choice == "put") {
            this.price = addon.european_call(this.s0, this.X, this.T, this.r, this.sigma, tao);
        }
        console.log(this.price)
        return this;
    }
    shoutCall(num_steps, shout) {
        this.price = addon.shout_call(this.s0, this.X, this.T, this.r, this.sigma, num_steps, shout);
        console.log(this.price)
        return this;
    }
    binaryOptions( num_simulations, fixedPayoff) {
        this.price = addon.monte_carlo_binary_options(this.s0, this.T, this.r, this.sigma, this.X, fixedPayoff, num_simulations);
        console.log(this.price)
        return this;
    }
    asianOptions(num_simulations, num_steps) {
        this.price = addon.asian_options(this.s0, this.X, this.T, this.r, this.sigma, num_simulations, num_steps);
        console.log(this.price)
        return this;
    }
    upAndOutCall(numSimulations, barrier) {
        this.price = addon.up_and_out_call(this.s0, this.X, this.T, this.r, this.sigma, numSimulations, barrier);
        console.log(this.price)
        return this;
    }



}




module.exports = ExoticOptions;



