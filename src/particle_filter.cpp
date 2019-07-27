/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // Set the number of particles
  num_particles = 100;

  // Noise Generator
  std::default_random_engine gen;
  std::normal_distribution<double> dist_x(0, std[0]);
  std::normal_distribution<double> dist_y(0, std[1]);
  std::normal_distribution<double> dist_theta(0, std[2]);

  // Initialize all particles
  // Add random Gaussian noise to each particle
  for (int i = 0; i < num_particles; ++i) {
    Particle particle;

    particle.id = i;
    particle.x = x + dist_x(gen);
    particle.y = y + dist_y(gen);
    particle.theta = theta + dist_theta(gen);
    particle.weight = 1.0;

    particles.push_back(particle);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
  // Noise Generator
  std::default_random_engine gen;
  std::normal_distribution<double> dist_x(0, std_pos[0]);
  std::normal_distribution<double> dist_y(0, std_pos[1]);
  std::normal_distribution<double> dist_theta(0, std_pos[2]);

  // Add measurements to each particle
  for (auto& particle : particles) {
    Particle prev = particle;

    // Motion Model
    if (yaw_rate == 0) {
      particle.x = prev.x + delta_t * velocity * cos(prev.theta);
      particle.y = prev.y + delta_t * velocity * sin(prev.theta);
      particle.theta = prev.theta;
    } else {
      const auto v_y = velocity / yaw_rate;
      const auto d_theta = yaw_rate * delta_t;
      const auto new_theta = prev.theta + d_theta;

      particle.x = prev.x + v_y * (sin(new_theta) - sin(prev.theta));
      particle.y = prev.y + v_y * (-cos(new_theta) + cos(prev.theta));
      particle.theta = new_theta;
    }

    // Add random Gaussian noise
    particle.x += dist_x(gen);
    particle.y += dist_y(gen);
    particle.theta += dist_theta(gen);
  }
}

std::shared_ptr<Map::single_landmark_s> findNearestLandmark(
    const Map& map_landmarks, const LandmarkObs& map_obs) {
  std::vector<double> dist_list;

  for (const auto& l : map_landmarks.landmark_list) {
    const double dist = pow(l.x_f - map_obs.x, 2) + pow(l.y_f - map_obs.y, 2);
    dist_list.push_back(dist);
  }

  const auto itr = std::min_element(std::begin(dist_list), std::end(dist_list));
  const auto idx = std::distance(std::begin(dist_list), itr);
  const auto nearest_landmark = std::make_shared<Map::single_landmark_s>(
      map_landmarks.landmark_list.at(idx));

  return nearest_landmark;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const std::vector<LandmarkObs>& observations,
                                   const Map& map_landmarks) {
  // Update the weights of each particle using a mult-variate Gaussian
  for (auto& particle : particles) {
    // Convert VEHICLE'S coordinate system to MAP'S coordinate system
    std::vector<LandmarkObs> map_observations;
    for (const auto& veh_obs : observations) {
      LandmarkObs map_obs;

      const auto theta = particle.theta;
      map_obs.x = veh_obs.x * cos(theta) - veh_obs.y * sin(theta) + particle.x;
      map_obs.y = veh_obs.x * sin(theta) + veh_obs.y * cos(theta) + particle.y;

      map_observations.push_back(map_obs);
    }

    // Calculate likelihood by mult-variate Gaussiandistribution
    std::vector<double> likes;
    for (const auto& map_obs : map_observations) {
      const auto sigma_x = std_landmark[0];
      const auto sigma_y = std_landmark[1];
      const auto cov_det = pow(sigma_x, 2) + pow(sigma_y, 2);

      // Assume X and Y is independent
      const auto landmark = findNearestLandmark(map_landmarks, map_obs);
      if (!landmark) {
        continue;
      }

      const auto tmp_x = pow(map_obs.x - landmark->x_f, 2) / pow(sigma_x, 2);
      const auto tmp_y = pow(map_obs.y - landmark->y_f, 2) / pow(sigma_y, 2);
      const auto like = exp(-(tmp_x + tmp_y) / 2) / sqrt(2 * M_PI * cov_det);

      likes.push_back(like);
    }

    particle.weight =
        std::accumulate(std::begin(likes), std::end(likes), 0.0) / likes.size();
  }

  // Create weights
  weights.clear();
  for (const auto& particle : particles) {
    weights.push_back(particle.weight);
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional
   *   to their weight.
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

  // Random Integers Generator
  std::default_random_engine gen;
  std::discrete_distribution<int> dist(std::begin(weights), std::end(weights));

  // Resample particles with replacement with probability proportional
  std::vector<Particle> resample_particles;
  for (int i = 0; i < num_particles; ++i) {
    resample_particles.push_back(particles[dist(gen)]);
  }

  particles = resample_particles;
}

void ParticleFilter::SetAssociations(Particle& particle,
                                     const std::vector<int>& associations,
                                     const std::vector<double>& sense_x,
                                     const std::vector<double>& sense_y) {
  // particle: the particle to which assign each listed association,
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

std::string ParticleFilter::getAssociations(Particle best) {
  std::vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  std::string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}

std::string ParticleFilter::getSenseCoord(Particle best, std::string coord) {
  std::vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  std::string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
