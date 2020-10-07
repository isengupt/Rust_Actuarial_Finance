function js_bermudanCall(curr_stock_price,exercise_price,maturity_date,risk_free_rate,volatility, num_steps, early_dates) {
    let dT = maturity_date / num_steps;
    
    let a = Math.exp((risk_free_rate * dT));
    
    let u =  Math.exp((volatility * Math.sqrt(dT)));
    
    let d = 1.0 / u;
    
    let p = (a - d) / (u - d);
    let mat = []
    for (let step = 1; step < num_steps +2 ; step++) {
        const filled = new Array(step).fill(0);
        mat.push(filled);
    }


    for (let j = 0; j < num_steps +1 ; j++) {
      mat[num_steps][j] = Math.max(curr_stock_price * (u**j) * (d**(num_steps - j)) - exercise_price, 0.0);
    }

    for (let i = num_steps -1; i >= 0 ; i--) {

        for (let j = 0; j < i + 1; j++) {
   
            let v1 = Math.exp(-risk_free_rate * dT) * (p * mat[i+ 1][j + 1] + (1.0 - p) * mat[i +1][j]);
            let v2 = 0.0;
            for (val of early_dates) {
                if (Math.abs(j * dT - val) < 0.01) {
                    v2 = Math.max((mat[i][j] - exercise_price), 0.0);
                }
                else {
                    v2 = 0.0;
                } 

            }
            mat[i][j] = Math.max(v1, v2);
        }
      }
      console.log(mat[0][0])
      return mat[0][0];
   
}

function pi_simulation(num_points) { 

    let x = []
    let y = []
    for (let j = 0; j < num_points ; j++) {
       x.push(Math.random())
       y.push(Math.random())
    }
    

    let in_circle = []
    for (let i = 0; i < num_points ; i++) {
        let coord = (x[i]**2 + y[i]**2);
        if (coord <= 1.0) {
            in_circle.push(coord);
        } 
     }


let our_pi = (in_circle.length) * (4.0) / num_points;
return our_pi
}

module.exports = {
    js_bermudanCall,
    pi_simulation
}