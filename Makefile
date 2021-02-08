save-libs:
	pip freeze > requirements.txt

clean-raw:
	rm -r ./data/raw/*

clean-corpus:
	rm -r ./data/corpus/*

