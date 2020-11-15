import xlrd


f = open("routeKyoto.txt", mode='w', encoding="utf-8", )
file = 'exp_map.xlsx'
wb = xlrd.open_workbook(filename=file)

sheet = wb.sheet_by_index(0)

for i in range(497):
    if sheet.cell(i,0).value:
        if i != 496:
            f.write(str(int(sheet.cell(i,0).value)) + ',' + str(int(sheet.cell(i,1).value)) + ',' + str(float(sheet.cell(i,4).value)) + ',' + str(int(sheet.cell(i,3).value)) + ',' + str(int(sheet.cell(i,5).value)) + ',' + str(int(sheet.cell(i,6).value)) + ',' + str(float(sheet.cell(i,7).value))+ '\n')
        else:
            f.write(str(int(sheet.cell(i,0).value)) + ',' + str(int(sheet.cell(i,1).value)) + ',' + str(float(sheet.cell(i,4).value)) + ',' + str(int(sheet.cell(i,3).value)) + ',' + str(int(sheet.cell(i,5).value)) + ',' + str(int(sheet.cell(i,6).value)) + ',' + str(float(sheet.cell(i,7).value)))
        # f.write(str(int(sheet.cell(i,1).value)) + ',' + str(int(sheet.cell(i,0).value)) + ',' + str(int(sheet.cell(i,4).value)) + ',' + str(int(sheet.cell(i,3).value)) + '\n')

f.close()