use neon::prelude::*;
use std::iter::FromIterator;
use std::cmp;
extern crate statrs;
extern crate rand;

use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rand::distributions::Distribution;
use statrs::distribution::{Normal, Univariate, Continuous, Uniform, DiscreteUniform};
use crate::statrs::distribution::InverseCDF;



fn js_pv_bond(mut cx: FunctionContext) -> JsResult<JsNumber>{
    let future_val = cx.argument::<JsNumber>(0)?.value() as f64;
    let rate = cx.argument::<JsNumber>(1)?.value() as f64;
    let num_periods = cx.argument::<JsNumber>(2)?.value() as i32;
    let cashflow = cx.argument::<JsNumber>(3)?.value() as f64;
    let pv = pv_annuity(cashflow, rate, num_periods) + present_value(future_val, rate, num_periods);
    println!("Present value bond : {:?}", pv);
    Ok(cx.number(pv))
   
}

fn pv_bond(future_val: f64, rate: f64, num_periods: i32,cashflow: f64) -> f64 {
    let pv = pv_annuity(cashflow, rate, num_periods) + present_value(future_val, rate, num_periods);
    println!("Present value bond : {:?}", pv);
    return pv;
}

fn js_present_value(mut cx: FunctionContext) -> JsResult<JsNumber>{
    let future_val = cx.argument::<JsNumber>(0)?.value() as f64;
    let rate = cx.argument::<JsNumber>(1)?.value() as f64;
    let num_periods = cx.argument::<JsNumber>(2)?.value() as i32;
    
    let pv = future_val / (1.0 + rate).powf(num_periods as f64);
   
    Ok(cx.number(pv))  
}


fn present_value(future_val: f64, rate: f64,num_periods: i32) -> f64 {
    let pv = future_val / (1.0 + rate).powf(num_periods as f64);
    println!("Present value: {:?}", pv);
    return pv;
}

fn js_pv_perpetuity(mut cx: FunctionContext) -> JsResult<JsNumber>{
    let payment = cx.argument::<JsNumber>(0)?.value() as f64;
    let rate = cx.argument::<JsNumber>(1)?.value() as f64;
    let perp = payment / rate;
    
    Ok(cx.number(perp))  
}

fn pv_perpetuity(payment: f64, rate: f64) {
    let perp = payment / rate;
    println!("Present value to perpetuity: {:?}", perp);
}

fn js_pv_perpetuity_due(mut cx: FunctionContext) -> JsResult<JsNumber>{
    let payment = cx.argument::<JsNumber>(0)?.value() as f64;
    let rate = cx.argument::<JsNumber>(1)?.value() as f64;
    let perp = payment / rate * (1.0 + rate);
    
    Ok(cx.number(perp))  
}

fn pv_perpetuity_due(payment: f64, rate: f64) {
    let perp = payment / rate * (1.0 + rate);
    println!("Present value to perpetuity: {:?}", perp);
}

fn js_pv_annuity(mut cx: FunctionContext) -> JsResult<JsNumber>{
    let payment = cx.argument::<JsNumber>(0)?.value() as f64;
    let rate = cx.argument::<JsNumber>(1)?.value() as f64;
    let num_periods = cx.argument::<JsNumber>(2)?.value() as i32;
    let pv = payment / rate * (1.0 - 1.0 / (1.0 +rate).powf(num_periods as f64));
    
    Ok(cx.number(pv))  
}

fn pv_annuity(payment: f64, rate: f64, num_periods: i32) -> f64 {
    let pv = payment / rate * (1.0 - 1.0 / (1.0 +rate).powf(num_periods as f64));
    println!("Present value annuity: {:?}", pv);
    return pv;
}

fn js_pv_annuity_due(mut cx: FunctionContext) -> JsResult<JsNumber>{
    let payment = cx.argument::<JsNumber>(0)?.value() as f64;
    let rate = cx.argument::<JsNumber>(1)?.value() as f64;
    let num_periods = cx.argument::<JsNumber>(2)?.value() as i32;
    let pv = payment / rate * (1.0 - 1.0 / (1.0 +rate).powf(num_periods as f64)) * (1.0 + rate);
    
    Ok(cx.number(pv))  
}

fn pv_annuity_due(payment: f64, rate: f64, num_periods: i32) {
    let pv = payment / rate * (1.0 - 1.0 / (1.0 +rate).powf(num_periods as f64)) * (1.0 + rate);
    println!("Present value annuity due: {:?}", pv);
}


fn js_pv_growing_annuity(mut cx: FunctionContext) -> JsResult<JsNumber>{
    let payment = cx.argument::<JsNumber>(0)?.value() as f64;
    let interest_rate = cx.argument::<JsNumber>(1)?.value() as f64;
    let num_periods = cx.argument::<JsNumber>(2)?.value() as i32;
    let growth_rate = cx.argument::<JsNumber>(3)?.value() as f64;
    let pv = payment / (interest_rate - growth_rate) * (1.0 - (1.0 + growth_rate).powf(num_periods as f64) / (1.0 + interest_rate).powf(num_periods as f64)); 
    
    Ok(cx.number(pv))  
}

fn pv_growing_annuity(payment: f64, interest_rate: f64, num_periods: i32, growth_rate: f64) {
    let pv = payment / (interest_rate - growth_rate) * (1.0 - (1.0 + growth_rate).powf(num_periods as f64) / (1.0 + interest_rate).powf(num_periods as f64)); 
    println!("Present value growing annuity: {:?}", pv);
}

fn js_future_value(mut cx: FunctionContext) -> JsResult<JsNumber>{
    let present_val = cx.argument::<JsNumber>(0)?.value() as f64;
    let rate = cx.argument::<JsNumber>(1)?.value() as f64;
    let num_periods = cx.argument::<JsNumber>(2)?.value() as i32;
    let fv = present_val * (1.0 + rate).powf(num_periods as f64);
    
    Ok(cx.number(fv))  
}

fn future_value(present_val: f64, rate: f64,num_periods: i32) {
    let fv = present_val * (1.0 + rate).powf(num_periods as f64);
    println!("Future value: {:?}", fv);
}

fn js_fv_annuity(mut cx: FunctionContext) -> JsResult<JsNumber>{
    let payment = cx.argument::<JsNumber>(0)?.value() as f64;
    let rate = cx.argument::<JsNumber>(1)?.value() as f64;
    let num_periods = cx.argument::<JsNumber>(2)?.value() as i32;
    let fv = payment / rate * ((1.0 + rate).powf(num_periods as f64) - 1.0);
    
    Ok(cx.number(fv))  
}

fn fv_annuity(payment: f64, rate: f64,num_periods: i32) {
    let fv = payment / rate * ((1.0 + rate).powf(num_periods as f64) - 1.0);
    println!("Future value annuity: {:?}", fv);
}

fn js_fv_annuity_due(mut cx: FunctionContext) -> JsResult<JsNumber>{
    let payment = cx.argument::<JsNumber>(0)?.value() as f64;
    let rate = cx.argument::<JsNumber>(1)?.value() as f64;
    let num_periods = cx.argument::<JsNumber>(2)?.value() as i32;
    let fv = payment / rate * ((1.0 + rate).powf(num_periods as f64) - 1.0) * (1.0 + rate);
    
    Ok(cx.number(fv))  
}


fn fv_annuity_due(payment: f64, rate: f64,num_periods: i32) {
    let fv = payment / rate * ((1.0 + rate).powf(num_periods as f64) - 1.0) * (1.0 + rate);
    println!("Future value annuity due : {:?}", fv);
}

fn js_fv_growing_annuity(mut cx: FunctionContext) -> JsResult<JsNumber>{
    let payment = cx.argument::<JsNumber>(0)?.value() as f64;
    let interest_rate = cx.argument::<JsNumber>(1)?.value() as f64;
    let num_periods = cx.argument::<JsNumber>(2)?.value() as i32;
    let growth_rate = cx.argument::<JsNumber>(3)?.value() as f64;
    let fv = payment / (interest_rate - growth_rate) * ((1.0 + interest_rate).powf(num_periods as f64) - (1.0 + growth_rate) * num_periods as f64);
    
    Ok(cx.number(fv))  
}

fn fv_growing_annuity(payment: f64, interest_rate: f64,num_periods: i32, growth_rate: f64) {
    let fv = payment / (interest_rate - growth_rate) * ((1.0 + interest_rate).powf(num_periods as f64) - (1.0 + growth_rate) * num_periods as f64);
    println!("Future value growing annuity: {:?}", fv);
}


fn js_net_present_val(mut cx: FunctionContext) -> JsResult<JsNumber> {
    let rate = cx.argument::<JsNumber>(0)?.value() as f64;
    let values_handle = cx.argument::<JsArray>(1)?;
    let values_vec: Vec<Handle<JsValue>> = values_handle.to_vec(&mut cx)?;
   
    let mut values: Vec<f64> = Vec::new();
    for (_, item) in values_vec.iter().enumerate() {
    let value = item.downcast::<JsNumber>().unwrap();
    values.push(value.value() as f64);
}
    let mut sum  = 0.0;
    for (index, v) in values.iter().enumerate() {
        sum = sum + v / (1.0 + rate).powf(index as f64);
    }
    println!("Net present value: {:?}", sum);
    Ok(cx.number(sum))  
}

fn net_present_val(rate: f64, values: &mut Vec<f64>) -> f64 {
    let mut sum  = 0.0;
    for (index, v) in values.iter().enumerate() {
        sum = sum + v / (1.0 + rate).powf(index as f64);
    }
    println!("Net present value: {:?}", sum);
    return sum
}

fn js_graph_npv(mut cx: FunctionContext) -> JsResult<JsObject> {
    let lower_range = cx.argument::<JsNumber>(1)?.value() as usize;
    let upper_range = cx.argument::<JsNumber>(2)?.value() as usize;
    let range = (lower_range, upper_range);
    let values_handle = cx.argument::<JsArray>(0)?;
    let values_vec: Vec<Handle<JsValue>> = values_handle.to_vec(&mut cx)?;
   
    let mut cashflows: Vec<f64> = Vec::new();
    for (_, item) in values_vec.iter().enumerate() {
    let value = item.downcast::<JsNumber>().unwrap();
    cashflows.push(value.value() as f64);
}
    
    

    let mut rates = Vec::<f64>::new();
    let mut npv = Vec::<f64>::new();
    for n in range.0..range.1 {

        rates.push(0.01 * n as f64);
        npv.push(net_present_val(0.01 * n as f64, &mut cashflows));
       
    }

    let js_rates = JsArray::new(&mut cx, rates.len() as u32);
    let js_npv = JsArray::new(&mut cx, npv.len() as u32);
    for (i, obj) in rates.iter().enumerate() {
        let js_string = cx.number(*obj as f64);
        js_rates.set(&mut cx, i as u32, js_string).unwrap();
    }

    for (i, obj) in npv.iter().enumerate() {
        let js_string = cx.number(*obj as f64);
        js_npv.set(&mut cx, i as u32, js_string).unwrap();
    }

    println!("{:?}", rates);
    println!("{:?}", npv);

    let js_object = JsObject::new(&mut cx);
    js_object.set(&mut cx, "x", js_rates)?;
    js_object.set(&mut cx, "y", js_npv)?;

    Ok(js_object)

}

fn graph_npv(cashflows: &mut Vec<f64>, range: (usize, usize)) {
    println!("{:?}", cashflows);
    let mut rates = Vec::<f64>::new();
    let mut npv = Vec::<f64>::new();
    for n in range.0..range.1 {
        rates.push(0.01 * n as f64);
        npv.push(net_present_val(0.01 * n as f64, cashflows));
       
    }
    println!("{:?}", rates);
    println!("{:?}", npv);


}

fn js_ytm(mut cx: FunctionContext) -> JsResult<JsNumber> {
    let face_value = cx.argument::<JsNumber>(0)?.value() as f64;
    let bond_price = cx.argument::<JsNumber>(1)?.value() as f64;
    let years = cx.argument::<JsNumber>(2)?.value() as i32;
    let ytm  = (face_value / bond_price).powf(1.0 / years as f64) - 1.0;

    Ok(cx.number(ytm as f64))
}

fn yield_to_maturity (face_value: f64, bond_price: f64, years: i32 ) {
    let ytm  = (face_value / bond_price).powf(1.0 / years as f64) - 1.0;
    println!("{:?}", ytm);
}

fn js_stock_price(mut cx: FunctionContext) -> JsResult<JsNumber> {
    let dividend = cx.argument::<JsNumber>(0)?.value() as f64;
    let sell_price = cx.argument::<JsNumber>(1)?.value() as f64;
    let rate = cx.argument::<JsNumber>(2)?.value() as f64;
    let price = (dividend + sell_price) / (1.0 + rate);

    Ok(cx.number(price as f64))
}

fn stock_price(dividend: f64, sell_price: f64, rate: f64) {
    let price = (dividend + sell_price) / (1.0 + rate);
    println!("{:?}", price);
}

fn js_two_per_stocks(mut cx: FunctionContext) -> JsResult<JsNumber> {
    
    let values_handle = cx.argument::<JsArray>(0)?;
    let values_vec: Vec<Handle<JsValue>> = values_handle.to_vec(&mut cx)?;
   
    let mut dividends: Vec<f64> = Vec::new();
    for (_, item) in values_vec.iter().enumerate() {
    let value = item.downcast::<JsNumber>().unwrap();
    dividends.push(value.value() as f64);
}
    let sell_price = cx.argument::<JsNumber>(1)?.value() as f64;
    let rate = cx.argument::<JsNumber>(2)?.value() as f64;
    
    let mut sum = 0.0;
    let last_val = dividends[dividends.len() -1];

    for n in 0..dividends.len() - 1 {
       sum = sum + (dividends[n] / (1.0 + rate)).powf(n as f64 +1.0 as f64);
    }

    sum = sum + (last_val + sell_price) / (1.0 + rate).powf(dividends.len() as f64);
    println!("{:?}", sum);

    Ok(cx.number(sum))
}

fn two_per_stocks(dividends: Vec<f64>, sell_price: f64, rate: f64 ) {
    let mut sum = 0.0;
    let last_val = dividends[dividends.len() -1];

    for n in 0..dividends.len() - 1 {
       sum = sum + (dividends[n] / (1.0 + rate)).powf(n as f64 +1.0 as f64);
    }

    sum = sum + (last_val + sell_price) / (1.0 + rate).powf(dividends.len() as f64);
    println!("{:?}", sum);
}

fn js_nperiod_model(mut cx: FunctionContext) -> JsResult<JsNumber> {
    
    let values_handle = cx.argument::<JsArray>(0)?;
    let values_vec: Vec<Handle<JsValue>> = values_handle.to_vec(&mut cx)?;
   
    let mut dividends: Vec<f64> = Vec::new();
    for (_, item) in values_vec.iter().enumerate() {
    let value = item.downcast::<JsNumber>().unwrap();
    dividends.push(value.value() as f64);
}
    let rate = cx.argument::<JsNumber>(1)?.value() as f64;
    let growth_rate = cx.argument::<JsNumber>(2)?.value() as f64;
    
    
    let last_val = dividends[dividends.len() -1];
    let mut cut_dividends = Vec::from_iter(dividends[0..dividends.len() - 1].iter().cloned());
    let p_n = (last_val) / (rate - growth_rate);
    let years = dividends.len() -1;

    let sum = net_present_val(rate, &mut cut_dividends) * (1.0 + rate) + pv_bond(p_n, rate, years as i32,0.0);

    println!("{:?}", sum);
    Ok(cx.number(sum))
}

fn pv_nperiod_model(dividends: Vec<f64>, rate: f64, growth_rate: f64 ) {
    let last_val = dividends[dividends.len() -1];
    let mut cut_dividends = Vec::from_iter(dividends[0..dividends.len() - 1].iter().cloned());
    let p_n = (last_val) / (rate - growth_rate);
    let years = dividends.len() -1;

    let sum = net_present_val(rate, &mut cut_dividends) * (1.0 + rate) + pv_bond(p_n, rate, years as i32,0.0);

    println!("{:?}", sum);
   /*  println!("{:?}", last_val);
    for n in 0..dividends.len() - 1 {
        println!("{:?}", dividends[n]);
        println!("{:?}", n);
       sum = sum + (dividends[n] / (1.0 + rate).powf(n as f64 + 1.0));
    }

    let Pn = (last_val) / (rate - growth_rate);
    println!("{:?}", Pn);

    println!("{:?}", sum); */
}

fn js_cost_of_equity(mut cx: FunctionContext) -> JsResult<JsNumber> {
    let pres_val = cx.argument::<JsNumber>(0)?.value() as f64;
    let future_val = cx.argument::<JsNumber>(1)?.value() as f64;
    let cap_gain_yield =  (future_val - pres_val) / pres_val;
    let div_yield = 1.0 / pres_val;
    let cost = cap_gain_yield + div_yield;

    println!("{:?}", cost);
    Ok(cx.number(cost))
}


fn cost_of_equity(pres_val: f64, future_val: f64) {
    let cap_gain_yield =  (future_val - pres_val) / pres_val;
    let div_yield = 1.0 / pres_val;
    let cost = cap_gain_yield + div_yield;

    println!("{:?}", cost);
}

fn js_convert_apr_rm(mut cx: FunctionContext) -> JsResult<JsNumber> {
    let percentage_rate = cx.argument::<JsNumber>(0)?.value() as f64;
    let comp_freq = cx.argument::<JsNumber>(1)?.value() as f64;
    let period_rate = cx.argument::<JsNumber>(2)?.value() as f64;
    let conversion = (1.0 + percentage_rate / comp_freq).powf(comp_freq / period_rate) -1.0;


    Ok(cx.number(conversion))
}

// interest conversion


fn convert_apr_rm(percentage_rate: f64,comp_freq: f64, period_rate: f64) {
    let conversion = (1.0 + percentage_rate / comp_freq).powf(comp_freq / period_rate) -1.0;
    println!("{:?}", conversion);

}

fn js_convert_apr_apr(mut cx: FunctionContext) -> JsResult<JsNumber> {
    let percentage_rate = cx.argument::<JsNumber>(0)?.value() as f64;
    let comp_freq = cx.argument::<JsNumber>(1)?.value() as f64;
    let period_rate = cx.argument::<JsNumber>(2)?.value() as f64;
    let conversion = ((1.0 + percentage_rate / comp_freq).powf(comp_freq / period_rate) -1.0) * period_rate;


    Ok(cx.number(conversion))
}

fn convert_apr_apr(percentage_rate: f64,comp_freq: f64, period_rate: f64) {
    let conversion = ((1.0 + percentage_rate / comp_freq).powf(comp_freq / period_rate) -1.0) * period_rate;
    println!("{:?}", conversion);
    
}

fn js_convert_apr_rc(mut cx: FunctionContext) -> JsResult<JsNumber> {
    let percentage_rate = cx.argument::<JsNumber>(0)?.value() as f64;
    let m = cx.argument::<JsNumber>(1)?.value() as f64;
  
    let num = m * (1.0 + percentage_rate / m).ln();
    Ok(cx.number(num))
}

fn apr_rc(percentage_rate: f64, m: f64) {
    let num = m * (1.0 + percentage_rate / m).ln();
    println!("{:?}", num);

}

fn js_convert_rc_rm(mut cx: FunctionContext) -> JsResult<JsNumber> {
    let rc = cx.argument::<JsNumber>(0)?.value() as f64;
    let m = cx.argument::<JsNumber>(1)?.value() as f64;
  
    let num = (rc / m).exp() - 1.0;
    Ok(cx.number(num))
}

fn rc_rm(rc: f64, m: f64) {
    let num = (rc / m).exp() - 1.0;
    println!("{:?}", num);

}

fn js_convert_rc_apr(mut cx: FunctionContext) -> JsResult<JsNumber> {
    let rc = cx.argument::<JsNumber>(0)?.value() as f64;
    let m = cx.argument::<JsNumber>(1)?.value() as f64;
  
    let num = m*((rc / m).exp() - 1.0);
    Ok(cx.number(num))
}

fn rc_apr(rc: f64, m: f64) {
    let num = m*((rc / m).exp() - 1.0);
    println!("{:?}", num);

}

fn js_binomial_american_call(mut cx: FunctionContext) -> JsResult<JsNumber> {
    let curr_stock_price = cx.argument::<JsNumber>(0)?.value() as f64;
    
    let exercise_price = cx.argument::<JsNumber>(1)?.value() as f64;
    
    let maturity_date = cx.argument::<JsNumber>(2)?.value() as f64;
    
    let risk_free_rate = cx.argument::<JsNumber>(3)?.value() as f64;
    
    let volatility = cx.argument::<JsNumber>(4)?.value() as f64;
    
    let num_steps = cx.argument::<JsNumber>(5)?.value() as i32;
    
    let dT = maturity_date as f64 / num_steps as f64;

    println!("dT {:?}", dT);
    let a = (risk_free_rate * dT).exp();
    println!("a {:?}", a);
    let u = (volatility * dT.sqrt()).exp();
    println!("u {:?}", u);
    let d = 1.0 / u;
    println!("d {:?}", d);
    let p = (a - d) / (u - d);
    println!("p {:?}", p);
    let mut mat : Vec<Vec<f64>> = Vec::new();
    for i in 1..num_steps + 2 {
        //println!("{:?}", i);
        let mut zero_vec = vec![0.0; i as usize];
       // println!("{:?}", zero_vec.len());
        mat.push(zero_vec);
    }
    for j in 0..num_steps+1{
       
        mat[num_steps as usize][j as usize] = (curr_stock_price * (u).powf(j as f64) * d.powf(num_steps  as f64 - j as f64) - exercise_price).max(0.0);
    }
   // println!("{:?}", mat);

    for i in (0..num_steps).rev() {
        //println!("{:?}", i);
        for j in (0..i + 1) {
            
            let v1 = (-risk_free_rate * dT).exp() * (p * mat[i as usize + 1 as usize][j as usize + 1 as usize] + (1.0 - p) * mat[i as usize +1 as usize][j as usize]);
            let v2 = (mat[i as usize][j as usize] - exercise_price).max(0.0);
            mat[i as usize][j as usize] = v1.max(v2);
        }
    }
    println!("{:?}", mat[0][0]);
    Ok(cx.number(mat[0][0] as f64))

}
fn binomial_american_call(curr_stock_price: f64, exercise_price: f64, maturity_date: f64, risk_free_rate: f64, volatility: f64, num_steps: i32) {
    let dT = maturity_date as f64 / num_steps as f64;
    println!("dT {:?}", dT);
    let a = (risk_free_rate * dT).exp();
    println!("a {:?}", a);
    let u = (volatility * dT.sqrt()).exp();
    println!("u {:?}", u);
    let d = 1.0 / u;
    println!("d {:?}", d);
    let p = (a - d) / (u - d);
    println!("p {:?}", p);
    let mut mat : Vec<Vec<f64>> = Vec::new();
    for i in 1..num_steps + 2 {
        //println!("{:?}", i);
        let mut zero_vec = vec![0.0; i as usize];
       // println!("{:?}", zero_vec.len());
        mat.push(zero_vec);
    }
    for j in 0..num_steps+1{
       
        mat[num_steps as usize][j as usize] = (curr_stock_price * (u).powf(j as f64) * d.powf(num_steps  as f64 - j as f64) - exercise_price).max(0.0);
    }
   // println!("{:?}", mat);

    for i in (0..num_steps).rev() {
        //println!("{:?}", i);
        for j in (0..i + 1) {
            
            let v1 = (-risk_free_rate * dT).exp() * (p * mat[i as usize + 1 as usize][j as usize + 1 as usize] + (1.0 - p) * mat[i as usize +1 as usize][j as usize]);
            let v2 = (mat[i as usize][j as usize] - exercise_price).max(0.0);
            mat[i as usize][j as usize] = v1.max(v2);
        }
    }
    println!("{:?}", mat[0][0])
   
}

fn js_bermudan_call(mut cx: FunctionContext) -> JsResult<JsNumber> {
    let curr_stock_price = cx.argument::<JsNumber>(0)?.value() as f64;
    let exercise_price = cx.argument::<JsNumber>(1)?.value() as f64;
    let maturity_date = cx.argument::<JsNumber>(2)?.value() as f64;
    let risk_free_rate = cx.argument::<JsNumber>(3)?.value() as f64;
    let volatility = cx.argument::<JsNumber>(4)?.value() as f64;
    let num_steps = cx.argument::<JsNumber>(5)?.value() as i32;

    let values_handle = cx.argument::<JsArray>(6)?;
    let values_vec: Vec<Handle<JsValue>> = values_handle.to_vec(&mut cx)?;
   
    let mut early_dates: Vec<f64> = Vec::new();
    for (_, item) in values_vec.iter().enumerate() {
    let value = item.downcast::<JsNumber>().unwrap();
    early_dates.push(value.value() as f64);
}
let dT = maturity_date as f64 / num_steps as f64;
    
    let a = (risk_free_rate * dT).exp();
    
    let u = (volatility * dT.sqrt()).exp();
    
    let d = 1.0 / u;
    
    let p = (a - d) / (u - d);
    
    let mut mat : Vec<Vec<f64>> = Vec::new();
    for i in 1..num_steps + 2 {
        //println!("{:?}", i);
        let mut zero_vec = vec![0.0; i as usize];
       // println!("{:?}", zero_vec.len());
        mat.push(zero_vec);
    }
    for j in 0..num_steps+1{
       
        mat[num_steps as usize][j as usize] = (curr_stock_price * (u).powf(j as f64) * d.powf(num_steps  as f64 - j as f64) - exercise_price).max(0.0);
    }
    

    for i in (0..num_steps).rev() {
        //println!("{:?}", i);
        for j in 0..i + 1 {
            
            let v1 = (-risk_free_rate * dT).exp() * (p * mat[i as usize + 1 as usize][j as usize + 1 as usize] + (1.0 - p) * mat[i as usize +1 as usize][j as usize]);
            let mut v2 = 0.0;
            for val in early_dates.iter() {
                if (j as f64 * dT - val).abs() < 0.01 {
                    v2 = (mat[i as usize][j as usize] - exercise_price).max(0.0);
                }
                else {
                    v2 = 0.0;
                } 

            }

            mat[i as usize][j as usize] = v1.max(v2);
            //let v2 = (mat[i as usize][j as usize] - exercise_price).max(0.0);
            //mat[i as usize][j as usize] = v1.max(v2);
        }
    }
   
    Ok(cx.number(mat[0][0] as f64))

}


fn bermudan_call(curr_stock_price: f64, exercise_price: f64, maturity_date: f64, risk_free_rate: f64, volatility: f64, num_steps: i32, early_dates: Vec<f64>) {
    let dT = maturity_date as f64 / num_steps as f64;
    println!("dT {:?}", dT);
    let a = (risk_free_rate * dT).exp();
    println!("a {:?}", a);
    let u = (volatility * dT.sqrt()).exp();
    println!("u {:?}", u);
    let d = 1.0 / u;
    println!("d {:?}", d);
    let p = (a - d) / (u - d);
    println!("p {:?}", p);
    let mut mat : Vec<Vec<f64>> = Vec::new();
    for i in 1..num_steps + 2 {
        //println!("{:?}", i);
        let mut zero_vec = vec![0.0; i as usize];
       // println!("{:?}", zero_vec.len());
        mat.push(zero_vec);
    }
    for j in 0..num_steps+1{
       
        mat[num_steps as usize][j as usize] = (curr_stock_price * (u).powf(j as f64) * d.powf(num_steps  as f64 - j as f64) - exercise_price).max(0.0);
    }
    println!("{:?}", mat);

    for i in (0..num_steps).rev() {
        //println!("{:?}", i);
        for j in 0..i + 1 {
            
            let v1 = (-risk_free_rate * dT).exp() * (p * mat[i as usize + 1 as usize][j as usize + 1 as usize] + (1.0 - p) * mat[i as usize +1 as usize][j as usize]);
            let mut v2 = 0.0;
            for val in early_dates.iter() {
                if (j as f64 * dT - val).abs() < 0.01 {
                    v2 = (mat[i as usize][j as usize] - exercise_price).max(0.0);
                }
                else {
                    v2 = 0.0;
                } 

            }

            mat[i as usize][j as usize] = v1.max(v2);
            //let v2 = (mat[i as usize][j as usize] - exercise_price).max(0.0);
            //mat[i as usize][j as usize] = v1.max(v2);
        }
    }
    println!("{:?}", mat[0][0])
}

fn js_european_put(mut cx: FunctionContext) -> JsResult<JsNumber> {
    let S = cx.argument::<JsNumber>(0)?.value() as f64;
    let X = cx.argument::<JsNumber>(1)?.value() as f64;
    let T = cx.argument::<JsNumber>(2)?.value() as f64;
    let r = cx.argument::<JsNumber>(3)?.value() as f64;
    let sigma = cx.argument::<JsNumber>(4)?.value() as f64;
    let tao = cx.argument::<JsNumber>(5)?.value() as f64;

    let d1=((S/X).ln() + r *T+0.5*sigma*sigma*tao)/(sigma*tao.sqrt());
    println!("{:?}", d1);
    let d2 = d1-sigma*((tao).sqrt());
    println!("{:?}", d2);
    let n = Normal::new(0.0, 1.0).unwrap();
    let put = X * (-r * T).exp() * n.cdf(-d2) - S * n.cdf(-d1);

    Ok(cx.number(put as f64))

}


fn european_put(S: f64,X: f64,T: f64,r: f64,sigma: f64,tao: f64) {
    let d1=((S/X).ln() + r *T+0.5*sigma*sigma*tao)/(sigma*tao.sqrt());
    println!("{:?}", d1);
    let d2 = d1-sigma*((tao).sqrt());
    println!("{:?}", d2);
    let n = Normal::new(0.0, 1.0).unwrap();
    let put = X * (-r * T).exp() * n.cdf(-d2) - S * n.cdf(-d1);
    println!("{:?}", put);
}

fn js_european_call(mut cx: FunctionContext) -> JsResult<JsNumber> {
    let S = cx.argument::<JsNumber>(0)?.value() as f64;
    let X = cx.argument::<JsNumber>(1)?.value() as f64;
    let T = cx.argument::<JsNumber>(2)?.value() as f64;
    let r = cx.argument::<JsNumber>(3)?.value() as f64;
    let sigma = cx.argument::<JsNumber>(4)?.value() as f64;
    let tao = cx.argument::<JsNumber>(5)?.value() as f64;

    let d1=((S/X).ln() + r *T+0.5*sigma*sigma*tao)/(sigma*tao.sqrt());
    println!("{:?}", d1);
    let d2 = d1-sigma*((tao).sqrt());
    println!("{:?}", d2);
    let n = Normal::new(0.0, 1.0).unwrap();
    let call = S * n.cdf(d1) - X * (-r * T).exp() * n.cdf(d2);
    println!("{:?}", call);

    Ok(cx.number(call as f64))

}

fn european_call(S: f64,X: f64,T: f64,r: f64,sigma: f64,tao: f64) {
    let d1=((S/X).ln() + r *T+0.5*sigma*sigma*tao)/(sigma*tao.sqrt());
    println!("{:?}", d1);
    let d2 = d1-sigma*((tao).sqrt());
    println!("{:?}", d2);
    let n = Normal::new(0.0, 1.0).unwrap();
    let call = S * n.cdf(d1) - X * (-r * T).exp() * n.cdf(d2);
    println!("{:?}", call);
}

fn js_shout_call(mut cx: FunctionContext) -> JsResult<JsNumber> {
    let curr_stock_price = cx.argument::<JsNumber>(0)?.value() as f64;
    let exercise_price = cx.argument::<JsNumber>(1)?.value() as f64;
    let maturity_date = cx.argument::<JsNumber>(2)?.value() as f64;
    let risk_free_rate = cx.argument::<JsNumber>(3)?.value() as f64;
    let volatility = cx.argument::<JsNumber>(4)?.value() as f64;
    let num_steps = cx.argument::<JsNumber>(5)?.value() as i32;
    let shout_level = cx.argument::<JsNumber>(6)?.value() as f64;

  
    let dT = maturity_date as f64 / num_steps as f64;
    println!("dT {:?}", dT);
    let a = (risk_free_rate * dT).exp();
    println!("a {:?}", a);
    let u = (volatility * dT.sqrt()).exp();
    println!("u {:?}", u);
    let d = 1.0 / u;
    println!("d {:?}", d);
    let p = (a - d) / (u - d);
    println!("p {:?}", p);
    let mut mat : Vec<Vec<f64>> = Vec::new();
    for i in 1..num_steps + 2 {
        //println!("{:?}", i);
        let mut zero_vec = vec![0.0; i as usize];
       // println!("{:?}", zero_vec.len());
        mat.push(zero_vec);
    }
    for j in 0..num_steps+1{
       
        mat[num_steps as usize][j as usize] = (curr_stock_price * (u).powf(j as f64) * d.powf(num_steps  as f64 - j as f64) - exercise_price).max(0.0);
    }
   // println!("{:?}", mat);

    for i in (0..num_steps).rev() {
        //println!("{:?}", i);
        for j in (0..i + 1) {
            
            let v1 = (-risk_free_rate * dT).exp() * (p * mat[i as usize + 1 as usize][j as usize + 1 as usize] + (1.0 - p) * mat[i as usize +1 as usize][j as usize]);
            let v2 = (mat[i as usize][j as usize] - shout_level).max(0.0);
            mat[i as usize][j as usize] = v1.max(v2);
        }
    }
    Ok(cx.number(mat[0][0] as f64))

}

fn shout_call(curr_stock_price: f64, exercise_price: f64, maturity_date: f64, risk_free_rate: f64, volatility: f64, num_steps: i32, shout_level: f64) {
    let dT = maturity_date as f64 / num_steps as f64;
    println!("dT {:?}", dT);
    let a = (risk_free_rate * dT).exp();
    println!("a {:?}", a);
    let u = (volatility * dT.sqrt()).exp();
    println!("u {:?}", u);
    let d = 1.0 / u;
    println!("d {:?}", d);
    let p = (a - d) / (u - d);
    println!("p {:?}", p);
    let mut mat : Vec<Vec<f64>> = Vec::new();
    for i in 1..num_steps + 2 {
        //println!("{:?}", i);
        let mut zero_vec = vec![0.0; i as usize];
       // println!("{:?}", zero_vec.len());
        mat.push(zero_vec);
    }
    for j in 0..num_steps+1{
       
        mat[num_steps as usize][j as usize] = (curr_stock_price * (u).powf(j as f64) * d.powf(num_steps  as f64 - j as f64) - exercise_price).max(0.0);
    }
   // println!("{:?}", mat);

    for i in (0..num_steps).rev() {
        //println!("{:?}", i);
        for j in (0..i + 1) {
            
            let v1 = (-risk_free_rate * dT).exp() * (p * mat[i as usize + 1 as usize][j as usize + 1 as usize] + (1.0 - p) * mat[i as usize +1 as usize][j as usize]);
            let v2 = (mat[i as usize][j as usize] - shout_level).max(0.0);
            mat[i as usize][j as usize] = v1.max(v2);
        }
    }
    println!("{:?}", mat[0][0])
}

fn js_terminal_stock_price(mut cx: FunctionContext) -> JsResult<JsNumber> {
    let S = cx.argument::<JsNumber>(0)?.value() as f64;
    let T = cx.argument::<JsNumber>(1)?.value() as f64;
    let r = cx.argument::<JsNumber>(2)?.value() as f64;
    let sigma = cx.argument::<JsNumber>(3)?.value() as f64;
    let mut randran = rand::thread_rng();
    let n = Normal::new(0.0, 1.0).unwrap();
  
    let tao = n.sample(&mut randran);
   // println!("{:?}", tao);
    //let terminalPrice=S * sp.exp((r - 0.5 * sigma**2)*T+sigma*sp.sqrt(T)*tao)
    let terminal_price = S * ((r - 0.5 * sigma.powf(2.0)) * T + sigma * T.sqrt() * tao).exp();

  
    Ok(cx.number(terminal_price))

}

fn terminal_stock_price(S: f64, T: f64, r: f64, sigma: f64) -> f64 {

    let mut randran = rand::thread_rng();
    let n = Normal::new(0.0, 1.0).unwrap();
  
    let tao = n.sample(&mut randran);
   // println!("{:?}", tao);
    //let terminalPrice=S * sp.exp((r - 0.5 * sigma**2)*T+sigma*sp.sqrt(T)*tao)
    let terminal_price = S * ((r - 0.5 * sigma.powf(2.0)) * T + sigma * T.sqrt() * tao).exp();
    //println!("{:?}", terminal_price);
    return terminal_price;
    
}

fn js_binary_call_payoff(mut cx: FunctionContext) -> JsResult<JsNumber> {
    let x = cx.argument::<JsNumber>(0)?.value() as f64;
    let sT = cx.argument::<JsNumber>(1)?.value() as f64;
    let payoff = cx.argument::<JsNumber>(2)?.value() as f64;
    
    let boolean = sT >= x;
    // println!("{:?}", boolean);
     let binary = match boolean {
         false => 0.0,
         true => payoff,
  
     };


  
    Ok(cx.number(binary))

}

fn binary_call_payoff(x: f64, sT: f64, payoff: f64) -> f64 {
   let boolean = sT >= x;
  // println!("{:?}", boolean);
   let binary = match boolean {
       false => 0.0,
       true => payoff,

   };
  //println!("{:?}", binary);
   return binary;
  /*   if (sT >= x) {
        println!("{:?}", payoff);
    } else {
        println!("{:?}", 0.0);
    } */

}

fn js_monte_carlo_binary_options(mut cx: FunctionContext) -> JsResult<JsNumber> {
    let S = cx.argument::<JsNumber>(0)?.value() as f64;
    let T = cx.argument::<JsNumber>(1)?.value() as f64;
    let r = cx.argument::<JsNumber>(2)?.value() as f64;
    let sigma = cx.argument::<JsNumber>(3)?.value() as f64;
    let x = cx.argument::<JsNumber>(4)?.value() as f64;
    let fixed_payoff = cx.argument::<JsNumber>(5)?.value() as f64;
    let num_simulations = cx.argument::<JsNumber>(6)?.value() as i32;
    let mut payoffs = 0.0;
    for _ in 0..num_simulations {
        let sT = terminal_stock_price(S, T, r, sigma);
        payoffs = payoffs + binary_call_payoff(x, sT, fixed_payoff);
    }
    let price = (-r * T).exp() * (payoffs / (num_simulations as f64));
    println!("{:?}", price);


  
    Ok(cx.number(price))

}

fn monte_carlo_binary_options(S : f64, T: f64,r: f64,sigma: f64, x: f64, fixed_payoff: f64, num_simulations: i32) {
    let mut payoffs = 0.0;
    for _ in 0..num_simulations {
        let sT = terminal_stock_price(S, T, r, sigma);
        payoffs = payoffs + binary_call_payoff(x, sT, fixed_payoff);
    }
    let price = (-r * T).exp() * (payoffs / (num_simulations as f64));
    println!("{:?}", price);


}

fn js_asian_options_arithmetic_average(mut cx: FunctionContext) -> JsResult<JsNumber> {
    let s0 = cx.argument::<JsNumber>(0)?.value() as f64;
    let x = cx.argument::<JsNumber>(1)?.value() as f64;
    let T = cx.argument::<JsNumber>(2)?.value() as f64;
    let r = cx.argument::<JsNumber>(3)?.value() as f64;
    let sigma = cx.argument::<JsNumber>(4)?.value() as f64;
   
    let num_simulations = cx.argument::<JsNumber>(5)?.value() as i32;
    let num_steps = cx.argument::<JsNumber>(6)?.value() as i32;
    

    let dt=T/(num_steps as f64);
    let mut call_vec = vec![0.0; num_simulations as usize];
    println!("{:?}", call_vec.len());
    for j in 0..num_simulations {
        //println!("{:?}", j);
        let mut sT = s0;
        let mut total = 0.0;
        for _ in 0..num_steps {
            let mut randran = rand::thread_rng();
            let n = Normal::new(0.0, 1.0).unwrap();
            let e = n.sample(&mut randran);
            sT = sT * ((r - 0.5 *sigma * sigma) * dt + sigma * e * dt.sqrt()).exp();
            total = total + sT;
            
        }
        let price_average = total / (num_steps as f64);
        //println!("{:?}", price_average);
        call_vec[j as usize] = (price_average-x).max(0.0);
    }
println!("{:?}", call_vec);

    let sum: f64 = call_vec.iter().sum();
    println!("{:?}", (sum as f64 / call_vec.len() as f64));
   
    let call_price = (sum as f64 / call_vec.len() as f64) * (-r * T).exp();
    println!("{:?}", call_price);
  
    Ok(cx.number(call_price))

}


fn asian_options_arithmetic_average(s0: f64, x: f64, T:f64, r: f64, sigma: f64, num_simulations:i32, num_steps: i32) {
    
    let dt=T/(num_steps as f64);
    let mut call_vec = vec![0.0; num_simulations as usize];
    println!("{:?}", call_vec.len());
    for j in 0..num_simulations {
        //println!("{:?}", j);
        let mut sT = s0;
        let mut total = 0.0;
        for _ in 0..num_steps {
            let mut randran = rand::thread_rng();
            let n = Normal::new(0.0, 1.0).unwrap();
            let e = n.sample(&mut randran);
            sT = sT * ((r - 0.5 *sigma * sigma) * dt + sigma * e * dt.sqrt()).exp();
            total = total + sT;
            
        }
        let price_average = total / (num_steps as f64);
        //println!("{:?}", price_average);
        call_vec[j as usize] = (price_average-x).max(0.0);
    }
println!("{:?}", call_vec);

    let sum: f64 = call_vec.iter().sum();
    println!("{:?}", (sum as f64 / call_vec.len() as f64));
   
    let call_price = (sum as f64 / call_vec.len() as f64) * (-r * T).exp();
    println!("{:?}", call_price);



}

fn js_bs_call(mut cx: FunctionContext) -> JsResult<JsNumber> {
    let S = cx.argument::<JsNumber>(0)?.value() as f64;
    let X = cx.argument::<JsNumber>(1)?.value() as f64;
    let T = cx.argument::<JsNumber>(2)?.value() as f64;
    let r = cx.argument::<JsNumber>(3)?.value() as f64;
    let sigma = cx.argument::<JsNumber>(4)?.value() as f64;
    let d1=((S/X).ln()+(r+sigma*sigma/2.)*T)/(sigma*T.sqrt()); 
    let d2 = d1-sigma*T.sqrt();
    let n = Normal::new(0.0, 1.0).unwrap();
    let call = S*n.cdf(d1)-X*(-r*T).exp()*n.cdf(d2);
  
    Ok(cx.number(call))

}

fn js_delta_call(mut cx: FunctionContext) -> JsResult<JsNumber> {
    let S = cx.argument::<JsNumber>(0)?.value() as f64;
    let X = cx.argument::<JsNumber>(1)?.value() as f64;
    let T = cx.argument::<JsNumber>(2)?.value() as f64;
    let rf = cx.argument::<JsNumber>(3)?.value() as f64;
    let sigma = cx.argument::<JsNumber>(4)?.value() as f64;
    let d1=((S/X).ln()+(rf+sigma*sigma/2.0)*T)/(sigma*T.sqrt()); 

    let n = Normal::new(0.0, 1.0).unwrap();
    let call = S*n.cdf(d1);
  
    Ok(cx.number(call))

}

fn js_delta_put(mut cx: FunctionContext) -> JsResult<JsNumber> {
    let S = cx.argument::<JsNumber>(0)?.value() as f64;
    let X = cx.argument::<JsNumber>(1)?.value() as f64;
    let T = cx.argument::<JsNumber>(2)?.value() as f64;
    let rf = cx.argument::<JsNumber>(3)?.value() as f64;
    let sigma = cx.argument::<JsNumber>(4)?.value() as f64;
    let d1=((S/X).ln()+(rf+sigma*sigma/2.0)*T)/(sigma*T.sqrt()); 
    let n = Normal::new(0.0, 1.0).unwrap();
    let put = S*n.cdf(d1) - 1.0;
  
    Ok(cx.number(put))

}


fn bs_call(S: f64,X: f64,T: f64,r: f64,sigma: f64) -> f64 {
    let d1=((S/X).ln()+(r+sigma*sigma/2.)*T)/(sigma*T.sqrt()); 
    let d2 = d1-sigma*T.sqrt();
    let n = Normal::new(0.0, 1.0).unwrap();
    let call = S*n.cdf(d1)-X*(-r*T).exp()*n.cdf(d2);
    //println!("{:?}", call);
    return call;
}

fn js_implied_vol_binary(mut cx: FunctionContext) -> JsResult<JsObject> {
    let S = cx.argument::<JsNumber>(0)?.value() as f64;
    let X = cx.argument::<JsNumber>(1)?.value() as f64;
    let T = cx.argument::<JsNumber>(2)?.value() as f64;
    let r = cx.argument::<JsNumber>(3)?.value() as f64;
    let c = cx.argument::<JsNumber>(4)?.value() as f64;
    let mut k=1.0;
    let mut volLow=0.001;
    let mut volMid = 0.0;
    let mut volHigh=1.0;
    let cLow=bs_call(S,X,T,r,volLow);
    let cHigh=bs_call(S,X,T,r,volHigh);

    if cLow>c || cHigh<c {
      println!("Values not in range");
    } else {
    while k ==1.0 {
        let cLow=bs_call(S,X,T,r,volLow);
        let cHigh=bs_call(S,X,T,r,volHigh);
        volMid=(volLow+volHigh)/2.0;
        
        let cMid=bs_call(S,X,T,r,volMid);
        if (cHigh-cLow).abs() < 0.01 {
            k=2.0;
        }
        else if cMid>c {
            volHigh=volMid;
        }
        else {
            volLow=volMid;
        }
   
    }

    }

    let object = JsObject::new(&mut cx);
    let vol_low = cx.number(volLow as f64);
    let vol_mid = cx.number(volMid as f64);
    let vol_high = cx.number(volHigh as f64);
    object.set(&mut cx, "volLow", vol_low).unwrap();
    object.set(&mut cx, "volMid", vol_mid).unwrap();
    object.set(&mut cx, "volHigh", vol_high).unwrap();
    Ok(object)
}

fn js_implied_vol_call(mut cx: FunctionContext) -> JsResult<JsObject> {
    let S = cx.argument::<JsNumber>(0)?.value() as f64;
    let X = cx.argument::<JsNumber>(1)?.value() as f64;
    let T = cx.argument::<JsNumber>(2)?.value() as f64;
    let r = cx.argument::<JsNumber>(3)?.value() as f64;
    let c = cx.argument::<JsNumber>(4)?.value() as f64;
    let mut output = (0.0,0.0,0.0);
    for i in 0..200 {
        let sigma=0.005*(i as f64 + 1.0);
        let d1=((S/X).ln()+(r+sigma*sigma/2.0)*T)/(sigma*(T).sqrt());
        let d2 = d1-sigma*(T).sqrt();
        let n = Normal::new(0.0, 1.0).unwrap();
  
        let diff=c-(S*n.cdf(d1)-X*(-r*T).exp()*n.cdf(d2));
   
        if (diff).abs() <=0.01 {
            output = (i as f64,sigma, diff)
        }
    }

    println!("{:?}", output);
    let object = JsObject::new(&mut cx);
    let js_i = cx.number(output.0 as f64);
    let js_sigma = cx.number(output.1 as f64);
    let diff = cx.number(output.2 as f64);
    object.set(&mut cx, "index", js_i).unwrap();
    object.set(&mut cx, "sigma", js_sigma).unwrap();
    object.set(&mut cx, "diff", diff).unwrap();
    Ok(object)

}

fn js_implied_vol_put_min(mut cx: FunctionContext) -> JsResult<JsObject> {
    let S = cx.argument::<JsNumber>(0)?.value() as f64;
    let X = cx.argument::<JsNumber>(1)?.value() as f64;
    let T = cx.argument::<JsNumber>(2)?.value() as f64;
    let r = cx.argument::<JsNumber>(3)?.value() as f64;
    let p = cx.argument::<JsNumber>(4)?.value() as f64;
    let mut implied_vol = 1.0;
    let mut min_value = 100.0;
    let mut k = 0.0;
    let mut put_out = 0.0;
    let output = (0.0,0.0,0.0);
    for i in 1..10000 {

        let sigma=0.0001*(i as f64 + 1.0);
        let d1=((S/X).ln()+(r+sigma*sigma/2.0)*T)/(sigma*(T).sqrt());
        let d2 = d1-sigma*(T).sqrt();
        let n = Normal::new(0.0, 1.0).unwrap();
  
        let put= X * (-r * T).exp() * n.cdf(-d2) - S * n.cdf(-d1);
        let abs_diff = (put - p).abs();
   
        if  abs_diff < min_value {
           let min_value = abs_diff;
           let implied_vol = sigma;
           let k = i as f64;
        }
        let put_out = put;
    }


    println!("{:?}", output);
    let object = JsObject::new(&mut cx);
    let js_k = cx.number(k as f64);
    let js_vol = cx.number(implied_vol as f64);
    let js_put = cx.number(put_out as f64);
    let js_min = cx.number(min_value as f64);
    object.set(&mut cx, "k", js_k).unwrap();
    object.set(&mut cx, "implied_volume", js_vol).unwrap();
    object.set(&mut cx, "put", js_put).unwrap();
    object.set(&mut cx, "min_val", js_min).unwrap();
    Ok(object)

}


fn js_up_and_out_call(mut cx: FunctionContext) -> JsResult<JsNumber> {
    let s0 = cx.argument::<JsNumber>(0)?.value() as f64;
    let x = cx.argument::<JsNumber>(1)?.value() as f64;
    let T = cx.argument::<JsNumber>(2)?.value() as f64;
    let r = cx.argument::<JsNumber>(3)?.value() as f64;
    let sigma = cx.argument::<JsNumber>(4)?.value() as f64;
    let num_simulations = cx.argument::<JsNumber>(5)?.value() as i32;
    let barrier= cx.argument::<JsNumber>(6)?.value() as f64;

    let num_steps=100;
    let dt=T / num_steps as f64;
    let mut total=0.0; 
    for j in 0..num_simulations { 
        let mut sT=s0;
        let mut out=false;
        for i in 0..num_steps { 
            let mut randran = rand::thread_rng();
            let n = Normal::new(0.0, 1.0).unwrap();
            let e = n.sample(&mut randran);
            sT = sT * ((r-0.5*sigma*sigma) * dt + sigma * e * dt.sqrt()).exp();
            if sT>barrier {
               out=true;
            } 
        }
        if out==false {
            total = total + bs_call(s0,x,T,r,sigma);
        }
    }
    let call =  total / num_simulations as f64; 
    Ok(cx.number(call))

}

fn up_and_out_call(s0: f64,x: f64 ,T: f64,r: f64,sigma: f64,num_simulations: i32,barrier: f64) {
    let num_steps=100;
    let dt=T / num_steps as f64;
    let mut total=0.0; 
    for j in 0..num_simulations { 
        let mut sT=s0;
        let mut out=false;
        for i in 0..num_steps { 
            let mut randran = rand::thread_rng();
            let n = Normal::new(0.0, 1.0).unwrap();
            let e = n.sample(&mut randran);
            sT = sT * ((r-0.5*sigma*sigma) * dt + sigma * e * dt.sqrt()).exp();
            if sT>barrier {
               out=true;
            } 
        }
        if out==false {
            total = total + bs_call(s0,x,T,r,sigma);
        }
    }
    let call =  total / num_simulations as f64; 
    println!("{:?}", call);
}

fn js_get_cdf(mut cx: FunctionContext) -> JsResult<JsNumber> {
    let x = cx.argument::<JsNumber>(0)?.value() as f64;
    let n = Normal::new(0.0, 1.0).unwrap();
    Ok(cx.number(n.cdf(x)))
}

fn get_cdf(x: f64) -> f64 {
    let n = Normal::new(0.0, 1.0).unwrap();
    return n.cdf(x)
}

fn js_get_percent_point(mut cx: FunctionContext) -> JsResult<JsNumber> {
    let cumulative_prob = cx.argument::<JsNumber>(0)?.value() as f64;
    let n = Normal::new(0.0, 1.0).unwrap();
    println!("{:?}", n.inverse_cdf(cumulative_prob));
    Ok(cx.number(n.inverse_cdf(cumulative_prob)))
}

fn get_percent_point(cumulative_prob: f64) -> f64{
    let n = Normal::new(0.0, 1.0).unwrap();
    println!("{:?}", n.inverse_cdf(cumulative_prob));
    return n.inverse_cdf(cumulative_prob);
}

fn js_KMV(mut cx: FunctionContext) -> JsResult<JsObject> {
    let equity = cx.argument::<JsNumber>(0)?.value() as f64;
    let debt = cx.argument::<JsNumber>(1)?.value() as f64;
    let T = cx.argument::<JsNumber>(2)?.value() as f64;
    let r = cx.argument::<JsNumber>(3)?.value() as f64;
    let sigma_e = cx.argument::<JsNumber>(4)?.value() as f64;

    let n=10000;
    let m=2000;
    let mut diffOld=(1.0_f64).powf(6.0); 
    let mut output = (0.0, 0.0, 0.0);
    for i in 1..10 {
        for j in 1..m {
            let A=equity+debt/2.0+(i as f64)*debt/(n as f64);
            let sigmaA=0.05+(j as f64)*(1.0-0.001)/(m as f64);
            let d1 = ((A/debt).ln()+(r+sigmaA*sigmaA/2.0)*T)/(sigmaA*(T).sqrt());
            let d2 = d1-sigmaA*(T).sqrt();
            let diff4A= (A*get_cdf(d1)-debt*(-r*T).exp()*get_cdf(d2)-equity)/A  ;
            let diff4sigmaE= A/equity*get_cdf(d1)*sigmaA-sigma_e;     
            let diffNew=(diff4A).abs()+(diff4sigmaE).abs();
            if diffNew<diffOld {
               diffOld=diffNew;
               output=(A,sigmaA,diffNew);
        
            }
        }
    }
    println!("{:?}", output);
    let object = JsObject::new(&mut cx);
    let js_A = cx.number(output.0 as f64);
    let js_sigma = cx.number(output.1 as f64);
    let diff = cx.number(output.2 as f64);
    object.set(&mut cx, "A", js_A).unwrap();
    object.set(&mut cx, "sigma", js_sigma).unwrap();
    object.set(&mut cx, "diff", diff).unwrap();
    Ok(object)


}
fn KMV(equity: f64,debt: f64,T: f64,r: f64,sigma_e: f64) {
    let n=10000;
    let m=2000;
    let mut diffOld=(1.0_f64).powf(6.0); 
    let mut output = (0.0, 0.0, 0.0);
    for i in 1..10 {
        for j in 1..m {
            let A=equity+debt/2.0+(i as f64)*debt/(n as f64);
            let sigmaA=0.05+(j as f64)*(1.0-0.001)/(m as f64);
            let d1 = ((A/debt).ln()+(r+sigmaA*sigmaA/2.0)*T)/(sigmaA*(T).sqrt());
            let d2 = d1-sigmaA*(T).sqrt();
            let diff4A= (A*get_cdf(d1)-debt*(-r*T).exp()*get_cdf(d2)-equity)/A  ;
            let diff4sigmaE= A/equity*get_cdf(d1)*sigmaA-sigma_e;     
            let diffNew=(diff4A).abs()+(diff4sigmaE).abs();
            if diffNew<diffOld {
               diffOld=diffNew;
               output=(A,sigmaA,diffNew);
        
            }
        }
    }
    println!("{:?}", output);
}

fn js_BIS_f(mut cx: FunctionContext) -> JsResult<JsArray> {
    let R0 = cx.argument::<JsNumber>(0)?.value() as f64;
    let s = cx.argument::<JsNumber>(1)?.value() as f64;
    let num_simulations = cx.argument::<JsNumber>(2)?.value() as i32;
    let mut randran = rand::thread_rng();
    let n = Uniform::new(0.0, 1.0).unwrap();
    let mut R = R0;
    //println!("{:?}", e);
    //let mut nums = Vec::<f64>::new();
    //let mut z = Vec::<f64>::new();
    let mut outputs = Vec::<f64>::new();
    for i in 0..num_simulations {
        let e = n.sample(&mut randran);
        //nums.push(e);
        //z.push(get_percent_point(e));
        let dR = get_percent_point(e) * (s / (2.0_f64).sqrt());
        let logR = R.ln();
        R = (logR + dR).exp();
        outputs.push(R);
    }

    println!("{:?}", outputs);

    let js_array = JsArray::new(&mut cx, outputs.len() as u32);
    // Iterate over the Rust Vec and map each value in the Vec to the JS array
    for (i, obj) in outputs.iter().enumerate() {
        let num = cx.number(*obj);
        js_array.set(&mut cx, i as u32, num).unwrap();
    }

    Ok(js_array)
    


}

fn BIS_f(R0: f64, s: f64, num_simulations: i32) {
    let mut randran = rand::thread_rng();
    let n = Uniform::new(0.0, 1.0).unwrap();
    let mut R = R0;
    //println!("{:?}", e);
    //let mut nums = Vec::<f64>::new();
    //let mut z = Vec::<f64>::new();
    let mut outputs = Vec::<f64>::new();
    for i in 0..num_simulations {
        let e = n.sample(&mut randran);
        //nums.push(e);
        //z.push(get_percent_point(e));
        let dR = get_percent_point(e) * (s / (2.0_f64).sqrt());
        let logR = R.ln();
        R = (logR + dR).exp();
        outputs.push(R);
    }

    println!("{:?}", outputs);
  
}

fn js_portfolio_variance(mut cx: FunctionContext) -> JsResult<JsObject> {
    let std1 = cx.argument::<JsNumber>(0)?.value() as f64;
    let std2 = cx.argument::<JsNumber>(1)?.value() as f64;
    let rho = cx.argument::<JsNumber>(2)?.value() as f64;

    let var1 = std1.powf(2.0);
    let var2 = std2.powf(2.0);
    let mut finalW1 = 0.0;

    let num_iterations = 1000;
    let mut init_variance = 10.0;
    let mut min_variance = 1.0 / (1000 as f64);

    for i in 0..num_iterations {
    let w1= (i as f64) * min_variance;
    let w2=1.0 - w1;
    let var=w1.powf(2.0) *var1 + w2.powf(2.0)* var2 +2.0 * w1 *w2 * rho * std1 *std2;
    if (var < init_variance) {
        init_variance = var;
        finalW1=w1;
        }
    }

    let js_object = JsObject::new(&mut cx);
    let js_variance = cx.number(init_variance as f64);
    let js_final = cx.number(finalW1 as f64);

    js_object.set(&mut cx, "min_vol", js_variance).unwrap();
    js_object.set(&mut cx, "w1", js_final).unwrap();

    Ok(js_object)
}

fn utility_function(mut cx: FunctionContext) -> JsResult<JsObject> {
    let a_close_handle = cx.argument::<JsArray>(0)?;
    let A = cx.argument::<JsNumber>(1)?.value() as i32;
    let a_close_vec: Vec<Handle<JsValue>> = a_close_handle.to_vec(&mut cx)?;
  
    let mut a_close: Vec<f64> = Vec::new();
    
    for (_, item) in a_close_vec.iter().enumerate() {
    let close_num = item.downcast::<JsNumber>().unwrap();
    a_close.push(close_num.value() as f64);
}

let ret1 = Vec::from_iter(a_close[1 as usize..(a_close.len() - 1) as usize].iter().cloned());
let ret2 = Vec::from_iter(a_close[0 as usize..(a_close.len() - 2) as usize].iter().cloned());
let mut returns = Vec::<f64>::new();
for j in 0..ret1.len() -1 {
    returns.push(ret1[j] / ret2[j] -1.0);
}

let meanDaily = average(&returns);
let varDaily = variance(&returns);

let meanAnnual = (1.0 + meanDaily ).powf(252.0);
let varAnnual = varDaily * 252.0;
let output = meanAnnual - 0.5 * A as f64 * varAnnual;

let js_object = JsObject::new(&mut cx);
let js_mean_annual = cx.number(meanAnnual as f64);
let js_var_annual = cx.number(varAnnual as f64);
let output = cx.number(output);

js_object.set(&mut cx, "mean_annual", js_mean_annual).unwrap();
js_object.set(&mut cx, "var_annual", js_var_annual).unwrap();
js_object.set(&mut cx, "output", output).unwrap();

Ok(js_object)



}

fn js_arch_model(mut cx: FunctionContext) -> JsResult<JsObject> {
    let num_observations = cx.argument::<JsNumber>(0)?.value() as i32;
    let padding = cx.argument::<JsNumber>(1)?.value() as i32;
 
    
    let n2 = num_observations + padding;
    let a = (0.1, 0.3);
    let mut randran = rand::thread_rng();
    let n = Normal::new(0.0, 1.0).unwrap();
    let mut errors = Vec::<f64>::new();
     for i in 0..n2 {
        let e = n.sample(&mut randran);
        errors.push(e);
    } 
   
    let mut t= vec![0.0; n2 as usize];

   // println!("{:?}", errors);
   // println!("{:?}", t);
    let mut rand2 = rand::thread_rng();
    let first_index = Normal::new(0.0, (a.0 as f64 / (1.0 - a.1) as f64).sqrt()).unwrap();
    t[0] = first_index.sample(&mut rand2);
    println!("{:?}", t[0]);
    for i in 1..(n2-1) {


       t[i as usize] = errors[i as usize] * (a.0 + a.1 * t[(i-1) as usize].powf(2.0)).sqrt();   
    }
    let y = Vec::from_iter(t[(padding-1) as usize..(t.len() - 1) as usize].iter().cloned());
    println!("{:?}", y);
    let x: Vec<i32> = (0..num_observations).collect();
    println!("{:?}", x);


    let js_x = JsArray::new(&mut cx, x.len() as u32);
    let js_y = JsArray::new(&mut cx, y.len() as u32);
    for (i, obj) in x.iter().enumerate() {
        let js_string = cx.number(*obj as i32);
        js_x.set(&mut cx, i as u32, js_string).unwrap();
    }

    for (i, obj) in y.iter().enumerate() {
        let js_string = cx.number(*obj as f64);
        js_y.set(&mut cx, i as u32, js_string).unwrap();
    }



    let js_object = JsObject::new(&mut cx);
    js_object.set(&mut cx, "x", js_x)?;
    js_object.set(&mut cx, "y", js_y)?;

    Ok(js_object)

}

fn arch_model(num_observations: i32, padding: i32) {
    let n2 = num_observations + padding;
    let a = (0.1, 0.3);
    let mut randran = rand::thread_rng();
    let n = Normal::new(0.0, 1.0).unwrap();
    let mut errors = Vec::<f64>::new();
     for i in 0..n2 {
        let e = n.sample(&mut randran);
        errors.push(e);
    } 
   
    let mut t= vec![0.0; n2 as usize];

   // println!("{:?}", errors);
   // println!("{:?}", t);
    let mut rand2 = rand::thread_rng();
    let first_index = Normal::new(0.0, (a.0 as f64 / (1.0 - a.1) as f64).sqrt()).unwrap();
    t[0] = first_index.sample(&mut rand2);
    println!("{:?}", t[0]);
    for i in 1..(n2-1) {


       t[i as usize] = errors[i as usize] * (a.0 + a.1 * t[(i-1) as usize].powf(2.0)).sqrt();   
    }
    let y = Vec::from_iter(t[(padding-1) as usize..(t.len() - 1) as usize].iter().cloned());
    println!("{:?}", y);
    let x: Vec<i32> = (0..num_observations).collect();
    println!("{:?}", x);

   
}

fn js_garch_model(mut cx: FunctionContext) -> JsResult<JsObject> {
    let num_observations = cx.argument::<JsNumber>(0)?.value() as i32;
    let padding = cx.argument::<JsNumber>(1)?.value() as i32;
 
    let n2 = num_observations + padding;
    let a = (0.1, 0.3);
    let alpha = (0.1, 0.3);
    let beta = 0.2;
    
    let mut randran = rand::thread_rng();
    let n = Normal::new(0.0, 1.0).unwrap();
    let mut errors = Vec::<f64>::new();
     for i in 0..n2 {
        let e = n.sample(&mut randran);
        errors.push(e);
    } 
   
    let mut t= vec![0.0; n2 as usize];

   // println!("{:?}", errors);
   // println!("{:?}", t);
    let mut rand2 = rand::thread_rng();
    let first_index = Normal::new(0.0, (a.0 as f64 / (1.0 - a.1) as f64).sqrt()).unwrap();
    t[0] = first_index.sample(&mut rand2);
    println!("{:?}", t[0]);
    for i in 1..(n2-1) {
       t[i as usize] = errors[i as usize] * (alpha.0 + alpha.1 * errors[(i-1) as usize].powf(2.0) + 2.0 + beta * t[(i-1) as usize].powf(2.0)).sqrt();
    }
    let y = Vec::from_iter(t[(padding-1) as usize..(t.len() - 1) as usize].iter().cloned());
    println!("{:?}", y);
    let x: Vec<i32> = (0..num_observations).collect();
    println!("{:?}", x);



    let js_x = JsArray::new(&mut cx, x.len() as u32);
    let js_y = JsArray::new(&mut cx, y.len() as u32);
    for (i, obj) in x.iter().enumerate() {
        let js_string = cx.number(*obj as i32);
        js_x.set(&mut cx, i as u32, js_string).unwrap();
    }

    for (i, obj) in y.iter().enumerate() {
        let js_string = cx.number(*obj as f64);
        js_y.set(&mut cx, i as u32, js_string).unwrap();
    }



    let js_object = JsObject::new(&mut cx);
    js_object.set(&mut cx, "x", js_x)?;
    js_object.set(&mut cx, "y", js_y)?;

    Ok(js_object)

}


fn garch_model(num_observations: i32, padding: i32) {
    let n2 = num_observations + padding;
    let a = (0.1, 0.3);
    let alpha = (0.1, 0.3);
    let beta = 0.2;
    
    let mut randran = rand::thread_rng();
    let n = Normal::new(0.0, 1.0).unwrap();
    let mut errors = Vec::<f64>::new();
     for i in 0..n2 {
        let e = n.sample(&mut randran);
        errors.push(e);
    } 
   
    let mut t= vec![0.0; n2 as usize];

   // println!("{:?}", errors);
   // println!("{:?}", t);
    let mut rand2 = rand::thread_rng();
    let first_index = Normal::new(0.0, (a.0 as f64 / (1.0 - a.1) as f64).sqrt()).unwrap();
    t[0] = first_index.sample(&mut rand2);
    println!("{:?}", t[0]);
    for i in 1..(n2-1) {
       t[i as usize] = errors[i as usize] * (alpha.0 + alpha.1 * errors[(i-1) as usize].powf(2.0) + 2.0 + beta * t[(i-1) as usize].powf(2.0)).sqrt();
    }
    let y = Vec::from_iter(t[(padding-1) as usize..(t.len() - 1) as usize].iter().cloned());
    println!("{:?}", y);
    let x: Vec<i32> = (0..num_observations).collect();
    println!("{:?}", x);

   
}



fn amihud_illiquidity(mut cx: FunctionContext)-> JsResult<JsNumber> {
    let a_close_handle = cx.argument::<JsArray>(0)?;
    let volumes_handle = cx.argument::<JsArray>(1)?;
    

    let a_close_vec: Vec<Handle<JsValue>> = a_close_handle.to_vec(&mut cx)?;
    let volumes_vec: Vec<Handle<JsValue>> = volumes_handle.to_vec(&mut cx)?;
    let mut a_close: Vec<f64> = Vec::new();
    let mut volumes: Vec<f64> = Vec::new();
    for (_, item) in a_close_vec.iter().enumerate() {
    let close_num = item.downcast::<JsNumber>().unwrap();
    a_close.push(close_num.value() as f64);
}


for (_, item) in volumes_vec.iter().enumerate() {
    let volume_num = item.downcast::<JsNumber>().unwrap();
    volumes.push(volume_num.value() as f64);
  }


 
    let mut dollar_vols = Vec::<f64>::new();
    for i in 0..a_close.len() - 1 {   
        dollar_vols.push(a_close[i] * volumes[i]);
    }
    println!("{:?}", dollar_vols);
    let ret1 = Vec::from_iter(a_close[1 as usize..(a_close.len() - 1) as usize].iter().cloned());
    let ret2 = Vec::from_iter(a_close[0 as usize..(a_close.len() - 2) as usize].iter().cloned());
    let mut returns = Vec::<f64>::new();
    for j in 0..ret1.len() -1 {
        returns.push((ret1[j] - ret2[j]) / ret1[j]);
    }
    let mut sum = 0.0;
    for i in 0..returns.len() -1 {
        sum = sum + (returns[i].abs() / dollar_vols[i]);
    }
    let illiq = sum / (returns.len() as f64);
    println!("{:?}", illiq);
    let number = cx.number(illiq);
    Ok(number)

}

fn average(numbers: &[f64]) -> f64 {
    numbers.into_par_iter().sum::<f64>() as f64 / numbers.len() as f64
}

fn variance(data: &[f64]) -> f64 {
    let count = data.len();
    let data_mean = average(&data);
            let variance = data.into_par_iter().map(|value| {
                let diff = data_mean - (*value as f64);

                diff * diff
            }).sum::<f64>() / count as f64;

           return variance
       
    
}

fn pi_simulation(mut cx: FunctionContext)-> JsResult<JsNumber> { 
    let num_points = cx.argument::<JsNumber>(0)?.value() as i32;
    let mut randran = rand::thread_rng();
    let n = Uniform::new(0.0, 1.0).unwrap();
    let mut x = Vec::<f64>::new();
     for i in 0..num_points as usize {
        let e = n.sample(&mut randran);
        x.push(e);
    } 
    let mut y = Vec::<f64>::new();
    for j in 0..num_points as usize {
       let e = n.sample(&mut randran);
       y.push(e);
   } 

   let mut dist = Vec::<f64>::new();
   let mut in_circle= Vec::<f64>::new();
   for i in 0..(num_points) as usize {
   let coord = (x[i].powf(2.0) + y[i].powf(2.0));
   if (coord <= 1.0) {
       in_circle.push(coord);
   } 
} 
let our_pi = (in_circle.len() as f64) * (4.0) / (num_points as f64);
Ok(cx.number(our_pi))
}

fn pick_random_stocks(mut cx: FunctionContext)-> JsResult<JsArray> { 
    let num_stocks = cx.argument::<JsNumber>(0)?.value() as i64;
    let num_stocks_provided = cx.argument::<JsNumber>(1)?.value() as i64;
    let mut randran = rand::thread_rng();
    let n = DiscreteUniform::new(1, num_stocks_provided).unwrap();
    let mut x = Vec::<i64>::new();
     for i in 0..num_stocks as usize {
        let e = n.sample(&mut randran) as i64;
        x.push(e);
    } 

    x.sort();
    x.dedup();

    let y= JsArray::new(&mut cx, x.len() as u32);

    for (i, obj) in x.iter().enumerate() {
        let js_string = cx.number(*obj as f64);
        y.set(&mut cx, i as u32, js_string).unwrap();
    }
 

 

Ok(y)
}







register_module!(mut cx, {
    // Present Value Functions
    cx.export_function("pv_bonds", js_pv_bond)?;
    cx.export_function("pv_f", js_present_value)?;
    cx.export_function("pv_perpetuity", js_pv_perpetuity)?;
    cx.export_function("pv_perpetuity_due", js_pv_perpetuity_due)?;
    cx.export_function("pv_annuity", js_pv_annuity)?;
    cx.export_function("pv_annuity_due", js_pv_annuity_due)?;
    cx.export_function("pv_growing_annuity", js_pv_growing_annuity)?;
    cx.export_function("npv", js_net_present_val)?;
    cx.export_function("get_npv_data", js_graph_npv)?;

    // Future Value Functions
    cx.export_function("fv_f", js_future_value)?;
    cx.export_function("fv_annuity", js_fv_annuity)?;
    cx.export_function("fv_annuity_due", js_fv_annuity_due)?;
    cx.export_function("fv_growing_annuity", js_fv_growing_annuity)?;
   
    //Yeild to Maturity
    cx.export_function("ytm", js_ytm)?;

    //Stock Price
    cx.export_function("stock_price", js_stock_price)?;
    cx.export_function("two_per_stocks", js_two_per_stocks)?;
    cx.export_function("get_nperiod_model_data", js_nperiod_model)?;
    cx.export_function("cost_of_equity", js_cost_of_equity)?;
    cx.export_function("terminal_stock_price", js_terminal_stock_price)?;
    cx.export_function("pick_random_stocks", pick_random_stocks)?;
    
    //Portfolio
    cx.export_function("portfolio_variance", js_portfolio_variance)?;
    cx.export_function("utility_function", utility_function)?;

    //APR
    cx.export_function("convert_apr_rm", js_convert_apr_rm)?;
    cx.export_function("convert_apr_apr", js_convert_apr_apr)?;
    cx.export_function("convert_apr_rc", js_convert_apr_rc)?;
    cx.export_function("convert_rc_rm", js_convert_rc_rm)?;
    cx.export_function("convert_rc_apr", js_convert_rc_apr)?;

    //Calls and Puts
    cx.export_function("binomial_american_call", js_binomial_american_call)?;
    cx.export_function("bermudan_call", js_bermudan_call)?;
    cx.export_function("european_call", js_european_put)?;
    cx.export_function("european_put", js_european_call)?;
    cx.export_function("shout_call", js_shout_call)?;
    cx.export_function("binary_call_payoff", js_binary_call_payoff)?;
    cx.export_function("bs_call", js_bs_call)?;
    cx.export_function("delta_call", js_delta_call)?;
    cx.export_function("delta_put", js_delta_put)?;
    cx.export_function("implied_vol_binary", js_implied_vol_binary)?;
    cx.export_function("implied_vol_call", js_implied_vol_call)?;
    cx.export_function("implied_vol_put", js_implied_vol_put_min)?;
    cx.export_function("up_and_out_call", js_up_and_out_call)?;
    cx.export_function("monte_carlo_binary_options", js_monte_carlo_binary_options)?;
    cx.export_function("asian_options", js_asian_options_arithmetic_average)?;


    //Auxilliary
    cx.export_function("get_cdf", js_get_cdf)?;
    cx.export_function("get_ppf", js_get_percent_point)?;

    //Graphs and Models
    cx.export_function("KMV", js_KMV)?;
    cx.export_function("BIS", js_BIS_f)?;
    cx.export_function("ARCH", js_arch_model)?;
    cx.export_function("GARCH", js_garch_model)?;
    cx.export_function("amihud_illiquidity", amihud_illiquidity)?;
    cx.export_function("pi_simulation", pi_simulation)?;


    Ok(())
});
