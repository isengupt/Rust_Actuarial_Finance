var addon = require("../native");

class APRConversion {
    constructor(percentage_rate, comp_freq, period_rate) {
        this.percentage_rate = percentage_rate;
        this.comp_freq = comp_freq;
        this.period_rate = period_rate;
        this.result =0.0;
    }
    APRtoRM() {
       this.result = addon.convert_apr_m(this.percentage_rate, this.comp_freq, this.period_rate);
       return this;
    }
    APRtoAPR() {
        this.result = addon.convert_apr_apr(this.percentage_rate, this.comp_freq, this.period_rate);
        return this;
    }
     APRtoRC(m) {
        this.result = addon.convert_apr_rc(this.percentage_rate, m);
        return this;
     }
     RCtoRM(rc, m) {
        this.result = addon.convert_apr_rc(rc, m);
        return this;
     }
     RCtoAPR(rc, m) {
        this.result = addon.convert_rc_apr(rc, m);
        return this;
     }

}

module.exports = APRConversion;