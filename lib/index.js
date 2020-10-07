const bench = require('./benchmarks');
const ExoticOptions = require('./ExoticOptions');
const Simulations = require('./models.js');
var Original = require('./originals.js');

let s0=40.0                 
let x=40.0                  
let T=6.0/12.0              
let r=0.05                
let sigma=0.2   
let tao =  1.0/12.0; 
let fixedPayoff = 10.0        
let num_simulations=1000     
let num_steps=100000 
let shout=(1+0.03)*s0
let T2=[3.0/12.0,4.0/12.0]
let n=1000                




const exotopts = new ExoticOptions(s0,x,T,r,sigma)
const simul = new Simulations(num_steps)

  console.log(
    'Neon:          ',
    bench(() => exotopts.bermudanCall(n, T2))
  );

  console.log(
    'Node:    ',
    bench(() => Original.js_bermudanCall(s0,x,T,r,sigma, 1000, T2))
  );


  console.log(
    'Neon:          ',
    bench(() => simul.EstimatePi())
  );

  console.log(
    'Node:    ',
    bench(() => Original.pi_simulation(num_steps))
  );






