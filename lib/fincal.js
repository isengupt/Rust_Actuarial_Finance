var addon = require("../native");

class FinCal {
    constructor(future_val, rate, num_periods) {
        this.future_val = future_val;
        this.rate = rate;
        this.num_periods = num_periods;
        this.result = 0.0;
    }
    pvBonds(cashflow) {
       this.result = addon.pv_bonds(this.future_val, this.rate, this.num_periods, cashflow);
       return this;
    }
    pv() {
        this.result = addon.pv_f(this.future_val, this.rate, this.num_periods);
        return this;
     }
     fv() {
        this.result = addon.fv_f(this.future_val, this.rate, this.num_periods);
        return this;
     }
    pvPerpetuity() {
        this.result = addon.pv_perpetuity(this.future_val, this.rate);
        return this;
     }
    pvPerpetuityDue() {
        this.result = addon.pv_perpetuity_due(this.future_val,this.rate);
        return this;
    }
    pvAnnuity() {
        this.result = addon.pv_annuity(this.future_val, this.rate, this.num_periods);
        return this;
     }

     fvAnnuity() {
        this.result = addon.fv_annuity(this.future_val, this.rate, this.num_periods);
        return this;
     }

     pvAnnuityDue() {
        this.result = addon.pv_annuity_due(this.future_val, this.rate, this.num_periods);
        return this;
     }

     fvAnnuityDue() {
        this.result = addon.fv_annuity_due(this.future_val, this.rate, this.num_periods);
        return this;
     }

     pvGrowingAnnuity(growth_rate) {
        this.result = addon.pv_growing_annuity(this.future_val, this.rate, this.num_periods, growth_rate);
        return this;
     }


     fcGrowingAnnuity(growth_rate) {
        this.result = addon.fv_growing_annuity(this.future_val, this.rate, this.num_periods, growth_rate);
        return this;
     }

}

module.exports = FinCal;