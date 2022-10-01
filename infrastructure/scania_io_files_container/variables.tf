variable "scania_resoure_group_name" {
  type    = string
  default = "scaniarg"
}

variable "scania_resoure_group_location" {
  type    = string
  default = "eastus"
}

variable "scania_storage_account_name" {
  type    = string
  default = "scaniasa"
}

variable "scania_account_tier" {
  type    = string
  default = "Standard"
}

variable "scania_account_replication_type" {
  type    = string
  default = "LRS"
}

variable "scania_storage_container_name" {
  type    = string
  default = "scania-io-files"
}

variable "scania_container_access_type" {
  type    = string
  default = "private"
}