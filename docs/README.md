## Generating the docs

Use [mkdocs](http://www.mkdocs.org/) structure to update the documentation.

Build locally with:

    mkdocs build

Serve locally with:

    mkdocs serve

# Euler Cluster Reminders

- Command is `ssh username@euler.ethz.ch`
- Transfer data [text](https://scicomp.ethz.ch/wiki/Storage_and_data_transfer) (use `scp`)
- To read a .out file, use `cat filename.out`

## Best practise

- tar locally -> scp/rsync to cluster -> untar onto local compute node
  `tar -cvf $HOME/folder.tar $HOME/folder`
  `scp $HOME/folder.tar euler.ethz.ch:/cluster/work/data`
  `tar -xvf :/cluster/work/data/folder.tar -C $TMPDIR`
