# Data Storage & Versioning

## Sample Data

While you may have your own project data to work with, for the purpose
of following through with this template guide, let's download
the sample data for the
[sample problem statement](./02-preface.md#guides-problem-statement)
at hand.

=== "Polyaxon VSCode Terminal"

    ```bash
    $ mkdir -p /polyaxon-v1-data/workspaces/<YOUR_NAME>/data
    $ wget https://storage.googleapis.com/aisg-mlops-pub-data/kapitan-hull/acl-movie-review-data-aisg.tar.gz
    $ tar -xzvf acl-movie-review-data-aisg.tar.gz
    $ rm acl-movie-review-data-aisg.tar.gz
    ```

In the following section, we will work towards processing the raw data
and eventually training a sentiment classifier model.
