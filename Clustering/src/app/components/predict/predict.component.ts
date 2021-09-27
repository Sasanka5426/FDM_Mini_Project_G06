import { Component} from '@angular/core';
import { FormBuilder, FormGroup, FormArray, FormControl, ValidatorFn } from '@angular/forms';
import { App } from '../../models/app';
import {FastApiService} from '../../services/fastapi.service';

@Component({
  selector: 'predict-component',
  templateUrl: './predict.component.html',
  styleUrls: ['./predict.component.css']
})

export class PredictComponent {
  // islands: string[] = ['Torgersen', 'Dream', 'Biscoe'];
  // sexes: string[] = ['FEMALE', 'MALE'];
  // spicies: string[] = ['Adelie', 'Chinstrap', 'Gentoo'];

  prediction = "";
 
  name="Pixel Draw - Number Art Coloring Book"
  app = new App(this.name);

  constructor(
    private _fastApiService: FastApiService,
  ) {
  }

  public onPredict(): void {
    
    this._fastApiService.predict(this.app).subscribe(
      response => this.prediction = response);
    console.log('scope is ' + this.prediction);
    
  }
}
