import argparse
from adalam import AdalamFilter


if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Match a colmap database with AdaLAM")
    p.add_argument("--database_path", "-d", required=True)
    p.add_argument("--image_pairs_path", "-i", required=True)
    opt = p.parse_args()
    matcher = AdalamFilter()
    matcher.match_colmap_database(database_path=opt.database_path, image_pairs_path=opt.image_pairs_path)
