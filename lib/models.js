var addon = require("../native");
var yahooFinance = require("yahoo-finance");

 

class Simulations {
  constructor(num_steps) {
    this.num_steps = num_steps;
  
  }

  EstimatePi() {
    return addon.pi_simulation(this.num_steps);
  }

  Illiquidity(symbols, from, to) {
    let liquidities = []
    yahooFinance.historical({
      symbols: symbols,
      from: from,
      to: to,
      // period: 'd'  // 'd' (daily), 'w' (weekly), 'm' (monthly), 'v' (dividends only)
    }, function (err, quotes) {

      for  (const [key, value] of Object.entries(quotes)) {
  
          let a_close = value.map(quote => quote.adjClose);
          let volumes = value.map(quote => quote.volume);
          console.log(a_close)
          console.log(volumes)
  
          liquidities.push({ticker: key, liquidities: addon.amihud_illiquidity(a_close, volumes)})
     }
      console.log(liquidities)
  
    }); 
    return liquidities
  }

  Garch(padding) {
    return addon.GARCH(this.num_steps, padding);
  }


}



var simul = new Simulations(10000);
let data = simul.Garch(100);
console.log(data)



module.exports = Simulations;
 

