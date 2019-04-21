for i in range(1, 36):
	print("./pathtracer -t 8 -l 4 -s 64 -m 8 -c ../cam/dragon-gif/{}.txt -f dragon_{}.png ../dae/sky/CBdragon_microfacet_au.dae".format(i, i))