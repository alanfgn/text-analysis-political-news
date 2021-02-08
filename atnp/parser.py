import atnp.utils as utils
import bs4
import csv
import os
import re
from readability.readability import Document
from atnp.corpus import RawCorpusReader

TEXT_TAGS = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li']
REMOVE_TAGS = ['button', 'script', 'img', 'picture', "i", "figcaption", "figure",
               "audio", "input", "map", "meta", "nav", "source", "style", "svg", "video", "track"]

CONTENT_CLASSES = ['content-media__description', "article__photo-gallery-teaser", "container-credits", "content-credits", "credits"
                   'article-date', "gallery__controls", "carousel-controls", "cont-img-aticle"]

NOT_IMPORTANT_TEXT = ["Leia também:", 'access_time', "Vídeo:", "Leia mais:", "LEIA MAIS:"
                      "VIDEO", 'Item Anterior', "Proximo Item", "Foto Anterior", "Proxima Foto", "LEIA TAMBÉM:"]

NOT_IMPORTANT_SENTENCES = r'^Acompanhe nas redes sociais|^\(Por|^Curta a nossa fanpage:|^Veja a íntegra da nota:|^VIDEO|^Foto Anterior|^Proxima Foto'


def is_dispensable(soup_element):
    if 'class' in soup_element.attrs and len(soup_element.attrs['class']) > 0 \
            and set(soup_element.attrs['class']).issubset(CONTENT_CLASSES):
        return True
    if 'itemprop' in soup_element.attrs and soup_element.attrs['itemprop'] == 'description':
        return True
    return False


def is_not_impotant_text(text):
    if text in NOT_IMPORTANT_TEXT:
        return True
    if re.match(NOT_IMPORTANT_SENTENCES, text) is not None:
        return True
    return False


def html_parser(html):
    html = Document(html).summary()
    soup = bs4.BeautifulSoup(html, 'lxml')

    for element in soup.find_all(REMOVE_TAGS):
        element.extract()

    for element in soup.find_all("div", {"class": CONTENT_CLASSES}):
        element.extract()

    return "\n\n".join([
        element.text for element in soup.find_all(TEXT_TAGS)
        if not is_dispensable(element) and not is_not_impotant_text(element.text)
    ])


def save_cat_file(target, rows):
    with open(os.path.join(target, "cat.csv"), "w", newline='\n') as file:
        writer = csv.writer(file, delimiter=' ')
        writer.writerows(rows)


def transform(dir_raws, dir_corpus, path_ignore, diferent_journal_encoding={"folha": ["iso-8859-1"]}):

    ignore_list = [fileid.strip()
                   for fileid in list(open(path_ignore, newline="\n"))]

    corpus = RawCorpusReader(dir_raws, r'[\w0-9-#_\.#+,]+\.json')

    print("%0*d Files " % (3, len(corpus.fileids())))

    utils.create_if_not_exists(dir_corpus)

    rows = []
    for file_json in corpus.raws():
        name = file_json["fileid"]
        if name in ignore_list:
            continue

        print("Transforming %s" % file_json["fileid"])

        labels = [name + ".txt", file_json['subject'], file_json['journal']]
        rows.append(labels)

        with open(os.path.join(dir_corpus, labels[0]), "w") as textfile:
            text = html_parser(file_json['html'])

            if file_json["journal"] in diferent_journal_encoding.keys():
                try_encoding = diferent_journal_encoding[file_json["journal"]]
             
                try:
                    text = text.encode(try_encoding).decode("utf-8", "ignore")
                except:  # NOSONAR
                    print("### Warning error on diferent encoding %s to %s ##" %
                          (try_encoding, file_json["journal"]))

            textfile.write(text)

    save_cat_file(dir_corpus, rows)
