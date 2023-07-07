# Fireball dataset

The dataset is massive once fully extracted.

```bash
wget -O fireball.tar.gz https://datasets.mechanus.zhu.codes/fireball-anonymized-nov-28-2022-kfdjl.tar.gz
tar -zxvf fireball.tar.gz && rm fireball.tar.gz
find anonymized/data -name '*.gz' -exec gzip -d {} \; #same for filtered
```