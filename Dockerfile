FROM ovlasovets/rpy2:latest

# Add a custom entrypoint script
RUN echo '#!/bin/bash' > /entrypoint.sh && \
    echo 'source /opt/python3_env/bin/activate' >> /entrypoint.sh && \
    echo 'mkdir -p /container/mount/point/data/results' >> /entrypoint.sh && \
    echo 'cd /container/mount/point || exit 1' >> /entrypoint.sh && \
    echo 'exec "$@"' >> /entrypoint.sh && \
    chmod +x /entrypoint.sh

# Set the custom entrypoint
ENTRYPOINT ["/entrypoint.sh"]
