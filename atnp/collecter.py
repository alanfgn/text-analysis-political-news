import atnp.utils as utils
import requests
import csv
import re
import os


def slice_url(url):
    match = re.search(utils.LINK_PATTERN, url)
    return match.group(1), match.group(2), match.group(3)


def gen_unique_name(domain, path):
    return "{}__{}".format(domain, path.replace("/", "_"))


def makerequest(row, header):
    url = row[header.index("url")]

    request = requests.get(url)
    _, domain, path = slice_url(url)

    print("[%d] %s" % (request.status_code, url))

    return {
        "fileid": gen_unique_name(domain, path),
        "url": url,
        "subject": row[header.index("subject")],
        "journal": row[header.index("journal")],
        "html": request.text
    }


def report(links_file, destination):

    print("Making report of file %s to %s" % (links_file, destination))
    utils.create_if_not_exists(destination)

    files = os.listdir(destination)
    links = csv.reader(open(links_file, newline="\n"), delimiter=',')
    header = next(links)

    lines = count_not_downloaded = count_dupl = file_abnormal = 0

    fileids = []

    for row in links:
        lines = lines + 1

        _, domain, path = slice_url(row[header.index("url")])
        fileid = gen_unique_name(domain, path) + ".json"

        if fileid not in files:
            count_not_downloaded = count_not_downloaded + 1
            print("[%s] Not Downloaded" % row[header.index("url")])

        if fileid in fileids:
            count_dupl = count_dupl + 1
            print("[%s] Duplicatet" % fileid)

        fileids.append(fileid)

    for filename in files:
        if filename not in fileids:
            file_abnormal = file_abnormal + 1
            print("[%s] Abnormal" % filename)

    print("\n########################\n")

    print("%0*d Lines in csv %s" % (3, lines, links_file))

    print("%0*d Files Downloaded" % (3, len(files)))
    print("%0*d Files not downloaded" % (3, count_not_downloaded))
    print("%0*d Files duplicated" % (3, count_dupl))
    print("%0*d Files abnormals" % (3, file_abnormal))


def download(links_file, destination):
    print("Making requests of file %s to %s" % (links_file, destination))

    with open(links_file, newline="\n") as links:
        reader = csv.reader(links, delimiter=',')
        header = next(reader)
        for row in reader:
            filejson = makerequest(row, header=header)
            utils.save_json(destination, filejson["fileid"], filejson)
