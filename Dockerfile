# Воспроизводимая сборка в окружении, как у check-code (ros:humble-ros-base).
#   docker build -t tram-backup-odometry .
#   docker run --rm -it -v /path/to/bags:/bags tram-backup-odometry
#   (внутри) bash ./src/tram_solution/scripts/run_check.sh /bags/<запись>
FROM ros:humble-ros-base
SHELL ["/bin/bash", "-c"]
WORKDIR /ws
COPY . /ws/src/tram_solution
# сборка не требует сети: зависимости — только пакеты ros-base
RUN source /opt/ros/humble/setup.bash && colcon build && \
    echo 'source /ws/install/setup.bash' >> /root/.bashrc
ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
