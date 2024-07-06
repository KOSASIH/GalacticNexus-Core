package main

import (
	"context"
	"log"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/s3"
)

type DevOpsService struct{}

func (s *DevOpsService) UploadFile(file []byte) error {
	sess, err := session.NewSession(&aws.Config{Region: aws.String("us-west-2")}, nil)
	if err!= nil {
		return err
	}
	s3Client := s3.New(sess)
	input := &s3.PutObjectInput{
		Bucket: aws.String("my-bucket"),
		Key:    aws.String("file.txt"),
		Body:  bytes.NewReader(file),
	}
	_, err = s3Client.PutObject(context.TODO(), input)
	return err
}
