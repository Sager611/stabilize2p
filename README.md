# 2p-stabilizer

Different approaches to stabilize 2-photon imaging video.

## Requirements

You need to install [Anaconda](https://docs.anaconda.com/anaconda/install/linux/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html#linux-installers).

Make sure that your `PATH` env variable includes the `bin/` directory of your conda installation. You can achieve this, in the case of Miniconda, by appending the following to `~/.bashrc`:

```bash
export PATH="$PATH:$HOME/miniconda3/bin"
```

## Installation

Run:

```shell
eval "$(conda shell.bash hook)"
conda activate
make install-env
```

Please note that `data/` should be a link pointing to the directory with the dataset. For example:

```shell
$ vdir -ph data/
total 2.5M
drwxrwxrwx 1 admin admin 256K Sep  4 11:23 200901_G23xU1/
drwxrwxrwx 1 admin admin 256K Sep  5 20:52 200908_G23xU1/
drwxrwxrwx 1 admin admin 256K Sep  6 05:04 200909_G23xU1/
drwxrwxrwx 1 admin admin 256K Sep  6 14:11 200910_G23xU1/
drwxrwxrwx 1 admin admin 256K Sep  7 17:37 200929_G23xU1/
drwxrwxrwx 1 admin admin 256K Sep  7 22:52 200930_G23xU1/
drwxrwxrwx 1 admin admin 256K Sep  8 02:19 201002_G23xU1/
-rwxrwxrwx 1 admin admin  442 Sep  4 11:23 copy.sh
-rwxrwxrwx 1 admin admin  173 Sep  9 18:18 du.txt
-rwxrwxrwx 1 admin admin  696 Sep  8 12:50 nohup.out
```
