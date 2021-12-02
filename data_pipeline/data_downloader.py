import os
import ssl
import datetime
import requests
import geopandas as gpd

def get_data_dict(df, route):
    key = 'cb50405b-9cce-440c-967c-09945d849519'
    service_dates = sorted(list(set(df['service_date'])))
    stops_dict = {}
    delinquent_stops_dict = {0:set(), 1:set()}
    service_date = service_dates[0]
    url = f'http://bustime.mta.info/api/where/stops-for-route/MTA%20NYCT_{route.upper()}.json?key={key}&includePolylines=false&time={service_date}&version=2'
    response = requests.get(url).json()
    response = response['data']['entry']['stopGroupings'][0]['stopGroups']
    assert len(response) == 2
    if response[0]['id'] == '1':
        stops_dict[0] = response[1]['stopIds']
        stops_dict[1] = response[0]['stopIds']
    else:
        assert response[0]['id'] == '0'
        stops_dict[0] = response[0]['stopIds']
        stops_dict[1] = response[1]['stopIds']
    for service_date in service_dates[1:]:
        url = f'http://bustime.mta.info/api/where/stops-for-route/MTA%20NYCT_{route.upper()}.json?key={key}&includePolylines=false&time={service_date}&version=2'
        response = requests.get(url).json()
        response = response['data']['entry']['stopGroupings'][0]['stopGroups']
        assert len(response) == 2
        if response[0]['id'] == '1':
            response_stops_dir_0 = response[1]['stopIds']
            response_stops_dir_1 = response[0]['stopIds']
        else:
            assert response[0]['id'] == '0'
            response_stops_dir_0 = response[0]['stopIds']
            response_stops_dir_1 = response[1]['stopIds']
        for i in range(len(stops_dict[0]), 0, -1):
            if stops_dict[0][0:i] == response_stops_dir_0[0:i]:
                stops_dict[0] = response_stops_dir_0[0:i]
                delinquent_stops_dict[0].update(response_stops_dir_0[i:])
                break
        for i in range(len(stops_dict[1]), 0, -1):
            if stops_dict[1][0:i] == response_stops_dir_1[0:i]:
                stops_dict[1] = response_stops_dir_1[0:i]
                delinquent_stops_dict[1].update(response_stops_dir_1[i:])
                break
    return stops_dict, delinquent_stops_dict  


def get_shipments(route, months, years):
    # solution to <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: certificate has expired (_ssl.c:1131)>
    # encountered at: gdf_temp = gpd.read_file(url)
    # source: https://moreless.medium.com/how-to-fix-python-ssl-certificate-verify-failed-97772d9dd14c 
    # not sure if this is kohser...
    if (not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None)):
        ssl._create_default_https_context = ssl._create_unverified_context
    gdf = None
    API_BASE_URL = 'http://api.buswatcher.org'

    if route not in mta_routes:
        print(f'{route} is not a valid route')
    else:
        print(route)
        for year in years:
            for month in months:
                for day in range(1,2):
                    if month == 2:
                        if year % 4 == 0:
                            if day > 29:
                                break
                        else:
                            if day > 28:
                                break
                    elif month in {4, 6, 9, 11}:
                        if day > 30:
                            break   
                    if datetime.date(year, month, day) >= datetime.date.today():
                        break
                    print(f'{month}/{day}/{year}')
                    for hour in range(0,24):
                        url = API_BASE_URL + f'/api/v2/nyc/{year}/{month}/{day}/{hour}/{route}/buses/geojson'
                        # try / except block for handling missing shipment
                        try:
                            gdf_temp = gpd.read_file(url)
                            print(f'\t{hour}')
                        except:
                            print(f'\t{hour} missing')
                            continue
                        if gdf is not None:
                            gdf = gdf.append(gdf_temp)
                        else:
                            gdf = gdf_temp
    return gdf

# not all of these are confirmed to be compatible with BusWatcher API (only B46, Bx12, and M15)
mta_routes = {
    'B2',
    'B3',
    'B4',
    'B6',
    'B7',
    'B8',
    'B9',
    'B11',
    'B12',
    'B13',
    'B14',
    'B15',
    'B16',
    'B17',
    'B20',
    'B24',
    'B25',
    'B26',
    'B31',
    'B32',
    'B35',
    'B36',
    'B37',
    'B38',
    'B39',
    'B41',
    'B42',
    'B43',
    'B44',
    'B44',
    'B45',
    'B46',
    'B46',
    'B47',
    'B48',
    'B49',
    'B52',
    'B54',
    'B57',
    'B60',
    'B61',
    'B62',
    'B63',
    'B64',
    'B65',
    'B67',
    'B68',
    'B69',
    'B70',
    'B74',
    'B82',
    'B82',
    'B83',
    'B84',
    'B100',
    'B103',
    'BM1',
    'BM2',
    'BM3',
    'BM4',
    'BM5',
    'Bx1',
    'Bx2',
    'Bx3',
    'Bx4',
    'Bx4A',
    'Bx5',
    'Bx6',
    'Bx6',
    'Bx7',
    'Bx8',
    'Bx9',
    'Bx10',
    'Bx11',
    'Bx12',
    'Bx12',
    'Bx13',
    'Bx15',
    'Bx16',
    'Bx17',
    'Bx18',
    'Bx19',
    'Bx20',
    'Bx21',
    'Bx22',
    'Bx23',
    'Bx24',
    'Bx26',
    'Bx27',
    'Bx28',
    'Bx29',
    'Bx30',
    'Bx31',
    'Bx32',
    'Bx33',
    'Bx34',
    'Bx35',
    'Bx36',
    'Bx38',
    'Bx39',
    'Bx40',
    'Bx41',
    'Bx41',
    'Bx42',
    'Bx46',
    'BxM1',
    'BxM2',
    'BxM3',
    'BxM4',
    'BxM6',
    'BxM7',
    'BxM8',
    'BxM9',
    'BxM10',
    'BxM11',
    'BxM18',
    'D99',
    'M1',
    'M2',
    'M3',
    'M4',
    'M5',
    'M7',
    'M8',
    'M9',
    'M10',
    'M11',
    'M12',
    'M14A',
    'M14D',
    'M15',
    'M15',
    'M20',
    'M21',
    'M22',
    'M23',
    'M31',
    'M34',
    'M34A',
    'M35',
    'M42',
    'M50',
    'M55',
    'M57',
    'M60',
    'M66',
    'M72',
    'M79',
    'M86',
    'M96',
    'M98',
    'M100',
    'M101',
    'M102',
    'M103',
    'M104',
    'M106',
    'M116',
    'Q1',
    'Q2',
    'Q3',
    'Q4',
    'Q5',
    'Q6',
    'Q7',
    'Q8',
    'Q9',
    'Q10',
    'Q11',
    'Q12',
    'Q13',
    'Q15',
    'Q15A',
    'Q16',
    'Q17',
    'Q18',
    'Q19',
    'Q20A',
    'Q20B',
    'Q21',
    'Q22',
    'Q23',
    'Q24',
    'Q25',
    'Q26',
    'Q27',
    'Q28',
    'Q29',
    'Q30',
    'Q31',
    'Q32',
    'Q33',
    'Q34',
    'Q35',
    'Q36',
    'Q37',
    'Q38',
    'Q39',
    'Q40',
    'Q41',
    'Q42',
    'Q43',
    'Q44',
    'Q46',
    'Q47',
    'Q48',
    'Q49',
    'Q50',
    'Q52',
    'Q53',
    'Q54',
    'Q55',
    'Q56',
    'Q58',
    'Q59',
    'Q60',
    'Q64',
    'Q65',
    'Q66',
    'Q67',
    'Q69',
    'Q70',
    'Q72',
    'Q76',
    'Q77',
    'Q83',
    'Q84',
    'Q85',
    'Q88',
    'Q100',
    'Q101',
    'Q102',
    'Q103',
    'Q104',
    'Q110',
    'Q111',
    'Q112',
    'Q113',
    'Q114',
    'QM1',
    'QM2',
    'QM3',
    'QM4',
    'QM5',
    'QM6',
    'QM7',
    'QM8',
    'QM10',
    'QM11',
    'QM12',
    'QM15',
    'QM16',
    'QM17',
    'QM18',
    'QM20',
    'QM21',
    'QM24',
    'QM25',
    'QM31',
    'QM32',
    'QM34',
    'QM35',
    'QM36',
    'QM40',
    'QM42',
    'QM44',
    'S40',
    'S42',
    'S44',
    'S46',
    'S48',
    'S51',
    'S52',
    'S53',
    'S54',
    'S55',
    'S56',
    'S57',
    'S59',
    'S61',
    'S62',
    'S66',
    'S74',
    'S76',
    'S78',
    'S79',
    'S81',
    'S84',
    'S86',
    'S89',
    'S90',
    'S91',
    'S92',
    'S93',
    'S94',
    'S96',
    'S98',
    'SIM1',
    'SIM1C',
    'SIM2',
    'SIM3',
    'SIM3C',
    'SIM4',
    'SIM4C',
    'SIM4X',
    'SIM5',
    'SIM6',
    'SIM7',
    'SIM8',
    'SIM8X',
    'SIM9',
    'SIM10',
    'SIM11',
    'SIM15',
    'SIM22',
    'SIM25',
    'SIM26',
    'SIM30',
    'SIM31',
    'SIM32',
    'SIM33',
    'SIM33C',
    'SIM34',
    'SIM35',
    'X27',
    'X28',
    'X37',
    'X38',
    'X63',
    'X64',
    'X68'
}

