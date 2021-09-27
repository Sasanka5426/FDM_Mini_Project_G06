import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
//import { TrainParams } from '../models/trainParams';
import { App } from '../models/app';

@Injectable({
  providedIn: 'root'
})
export class FastApiService {

  private _http: HttpClient;
  private _baseUrl: string;
 

  constructor(http: HttpClient) {
    this._http = http;
    this._baseUrl = "http://127.0.0.1:60525 "
   
  }

  // public train(trainParams: TrainParams): Observable<any> {
  //   return this._http.post<TrainParams>(this._baseUrl + '/train', trainParams);
  // }

  // public getAccuracy(): Observable<number> {
  //   return this._http.get<number>(this._baseUrl + '/accuracy');
  // }

  public predict(data: App): Observable<string> {
    //this.returned_data=this._http.post<string>("http://localhost:8000" + '/predict', data)
    return this._http.post<string>("http://localhost:8000" + '/predict', data);
  }
}
