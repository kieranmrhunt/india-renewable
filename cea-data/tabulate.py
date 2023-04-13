import tabula

tables = tabula.read_pdf("installed-capacity-raw.pdf", pages='all', multiple_tables=True, lattice=True)


print(len(tables))



for n,df in enumerate(tables):
	df.to_csv("cea-table-dump/{:04d}.csv".format(n), index=False)


